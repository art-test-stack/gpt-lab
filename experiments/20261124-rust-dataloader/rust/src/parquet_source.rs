use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;
use parquet::record::Field;
use parquet::schema::types::Type;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;
use std::time::Duration;

fn is_eio(error: &(dyn Error + 'static)) -> bool {
    #[cfg(unix)]
    if error
        .downcast_ref::<std::io::Error>()
        .is_some_and(|error| error.raw_os_error() == Some(libc::EIO))
    {
        return true;
    }
    error.source().is_some_and(is_eio)
}

fn read_group_with_retry<R, T>(
    reader: &mut Option<R>,
    mut open: impl FnMut() -> ParquetResult<R>,
    mut read: impl FnMut(&R) -> ParquetResult<T>,
    mut sleep: impl FnMut(Duration),
) -> ParquetResult<T> {
    for attempt in 0..5 {
        let result = (|| {
            if reader.is_none() {
                *reader = Some(open()?);
            }
            read(reader.as_ref().unwrap())
        })();
        match result {
            Ok(value) => return Ok(value),
            Err(error) if is_eio(&error) && attempt < 4 => {
                *reader = None;
                sleep(Duration::from_secs(1 << attempt));
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!()
}

pub type DocumentRef = (usize, usize, usize);
pub type CursorTuple = (usize, Option<usize>, usize, usize, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cursor {
    pub shard_idx: usize,
    pub global_shard_idx: Option<usize>,
    pub row_group_idx: usize,
    pub offset_in_row_group: usize,
    pub epoch: usize,
}

impl Default for Cursor {
    fn default() -> Self {
        Self {
            shard_idx: 0,
            global_shard_idx: None,
            row_group_idx: 0,
            offset_in_row_group: 0,
            epoch: 1,
        }
    }
}

impl From<CursorTuple> for Cursor {
    fn from(value: CursorTuple) -> Self {
        Self {
            shard_idx: value.0,
            global_shard_idx: value.1,
            row_group_idx: value.2,
            offset_in_row_group: value.3,
            epoch: value.4,
        }
    }
}

impl From<Cursor> for CursorTuple {
    fn from(value: Cursor) -> Self {
        (
            value.shard_idx,
            value.global_shard_idx,
            value.row_group_idx,
            value.offset_in_row_group,
            value.epoch,
        )
    }
}

pub struct TextSource {
    shards: Vec<(usize, PathBuf)>,
    column: String,
    training: bool,
    rank: usize,
    world_size: usize,
    cursor: Cursor,
    reader: Option<SerializedFileReader<File>>,
    texts: Option<Vec<String>>,
    had_documents: bool,
    resuming: bool,
    exhausted: bool,
}

impl TextSource {
    pub fn new(
        shards: Vec<(usize, PathBuf)>,
        column: String,
        training: bool,
        rank: usize,
        world_size: usize,
        cursor: Option<Cursor>,
    ) -> Result<Self, String> {
        if shards.is_empty() || world_size == 0 || rank >= world_size {
            return Err("Invalid shard assignment or rank topology".into());
        }
        let first_group = if training { 0 } else { rank };
        let stride = if training { 1 } else { world_size };
        let position = cursor.unwrap_or(Cursor {
            row_group_idx: first_group,
            ..Cursor::default()
        });
        if position.shard_idx >= shards.len()
            || position.epoch == 0
            || position.row_group_idx < first_group
            || (position.row_group_idx - first_group) % stride != 0
            || position
                .global_shard_idx
                .is_some_and(|id| id != shards[position.shard_idx].0)
        {
            return Err("Invalid data checkpoint cursor".into());
        }
        Ok(Self {
            shards,
            column,
            training,
            rank,
            world_size,
            cursor: position,
            reader: None,
            texts: None,
            had_documents: cursor.is_some(),
            resuming: cursor.is_some(),
            exhausted: false,
        })
    }

    fn open(&self, index: usize) -> ParquetResult<SerializedFileReader<File>> {
        SerializedFileReader::new(File::open(&self.shards[index].1)?)
    }

    fn read_group(
        &self,
        reader: &SerializedFileReader<File>,
        group: usize,
    ) -> ParquetResult<Vec<String>> {
        if group >= reader.num_row_groups() {
            return Err(ParquetError::General(format!(
                "Data checkpoint row group {group} is outside the shard"
            )));
        }
        let schema = reader.metadata().file_metadata().schema();
        let field = schema
            .get_fields()
            .iter()
            .find(|field| field.name() == self.column)
            .ok_or_else(|| ParquetError::General(format!("Missing text column {}", self.column)))?;
        let projection = Type::group_type_builder(schema.name())
            .with_fields(vec![field.clone()])
            .build()?;
        let group_reader = reader.get_row_group(group)?;
        let rows = group_reader.get_row_iter(Some(projection))?;
        rows.map(|row| match row?.into_columns().pop() {
            Some((_, Field::Str(text))) => Ok(text),
            _ => Err(ParquetError::General(format!(
                "Column {} must contain non-null UTF-8 text",
                self.column
            ))),
        })
        .collect()
    }

    fn load_group(
        &self,
        index: usize,
        group: usize,
        reader: &mut Option<SerializedFileReader<File>>,
    ) -> Result<Vec<String>, String> {
        read_group_with_retry(
            reader,
            || self.open(index),
            |reader| self.read_group(reader, group),
            std::thread::sleep,
        )
        .map_err(|error| {
            format!(
                "Failed to read {}, row group {group}, rank {}: {error}",
                self.shards[index].1.display(),
                self.rank,
            )
        })
    }

    pub fn next_chunk(&mut self, limit: usize) -> Result<Option<(Vec<String>, Cursor)>, String> {
        if limit == 0 {
            return Err("Shard iteration batch size must be positive".into());
        }
        while !self.exhausted {
            if self.reader.is_none() {
                self.reader = Some(self.open(self.cursor.shard_idx).map_err(|error| {
                    format!(
                        "Cannot open {}: {error}",
                        self.shards[self.cursor.shard_idx].1.display()
                    )
                })?);
                if self.resuming
                    && self.cursor.row_group_idx >= self.reader.as_ref().unwrap().num_row_groups()
                {
                    return Err("Data checkpoint row group is outside the shard".into());
                }
                self.resuming = false;
            }
            let reader = self.reader.as_ref().unwrap();
            if self.cursor.row_group_idx < reader.num_row_groups() {
                if self.texts.is_none() {
                    let mut reader = self.reader.take();
                    let result = self.load_group(
                        self.cursor.shard_idx,
                        self.cursor.row_group_idx,
                        &mut reader,
                    );
                    self.reader = reader;
                    let texts = result?;
                    if self.cursor.offset_in_row_group > texts.len() {
                        return Err("Data checkpoint offset is outside the row group".into());
                    }
                    self.texts = Some(texts);
                }
                let texts = self.texts.as_mut().unwrap();
                if self.cursor.offset_in_row_group < texts.len() {
                    let end = self
                        .cursor
                        .offset_in_row_group
                        .saturating_add(limit)
                        .min(texts.len());
                    let chunk = texts[self.cursor.offset_in_row_group..end]
                        .iter_mut()
                        .map(std::mem::take)
                        .collect();
                    self.cursor.offset_in_row_group = end;
                    self.cursor.global_shard_idx = Some(self.shards[self.cursor.shard_idx].0);
                    self.had_documents = true;
                    return Ok(Some((chunk, self.cursor)));
                }
                self.texts = None;
                self.cursor.offset_in_row_group = 0;
                self.cursor.row_group_idx += if self.training { 1 } else { self.world_size };
                continue;
            }
            self.reader = None;
            self.cursor.shard_idx += 1;
            self.cursor.row_group_idx = if self.training { 0 } else { self.rank };
            if self.cursor.shard_idx == self.shards.len() {
                if !self.training {
                    self.exhausted = true;
                    return Ok(None);
                }
                if !self.had_documents {
                    return Err("No training documents for this rank".into());
                }
                self.cursor.shard_idx = 0;
                self.cursor.epoch += 1;
                self.had_documents = false;
                self.resuming = false;
            }
        }
        Ok(None)
    }

    pub fn read_documents(&self, refs: &[DocumentRef]) -> Result<Vec<String>, String> {
        let mut groups = HashMap::new();
        for &(shard, group, _) in refs {
            if groups.contains_key(&(shard, group)) {
                continue;
            }
            let index = self
                .shards
                .iter()
                .position(|(id, _)| *id == shard)
                .ok_or_else(|| format!("Invalid checkpoint shard reference {shard}"))?;
            groups.insert((shard, group), self.load_group(index, group, &mut None)?);
        }
        refs.iter()
            .map(|&(shard, group, row)| {
                groups[&(shard, group)].get(row).cloned().ok_or_else(|| {
                    format!(
                        "Checkpoint document is missing at {:?}",
                        (shard, group, row)
                    )
                })
            })
            .collect()
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn transient_eio_reopens_and_preserves_resume_cursor() {
        let source = TextSource::new(
            vec![(7, PathBuf::from("unused.parquet"))],
            "text".into(),
            true,
            0,
            1,
            Some(Cursor::from((0, Some(7), 2, 3, 4))),
        )
        .unwrap();
        let committed = source.cursor;
        let mut reader = Some(0);
        let opens = Cell::new(0);
        let attempts = Cell::new(0);
        let mut delays = Vec::new();
        let texts = read_group_with_retry(
            &mut reader,
            || {
                opens.set(opens.get() + 1);
                Ok(opens.get())
            },
            |reader| {
                assert_eq!(source.cursor, committed);
                assert!(source.texts.is_none());
                attempts.set(attempts.get() + 1);
                if attempts.get() < 3 {
                    assert_eq!(*reader, attempts.get() - 1);
                    Err(std::io::Error::from_raw_os_error(libc::EIO).into())
                } else {
                    assert_eq!(*reader, 2);
                    Ok(vec!["first".to_owned(), "second".to_owned()])
                }
            },
            |delay| delays.push(delay.as_secs()),
        )
        .unwrap();
        assert_eq!(texts, ["first", "second"]);
        assert_eq!(attempts.get(), 3);
        assert_eq!(opens.get(), 2);
        assert_eq!(delays, [1, 2]);
        assert_eq!(source.cursor, committed);
    }

    #[test]
    fn persistent_eio_stops_after_five_attempts() {
        let mut reader = None;
        let opens = Cell::new(0);
        let attempts = Cell::new(0);
        let mut delays = Vec::new();
        let result: ParquetResult<()> = read_group_with_retry(
            &mut reader,
            || {
                opens.set(opens.get() + 1);
                Ok(())
            },
            |_| {
                attempts.set(attempts.get() + 1);
                Err(std::io::Error::from_raw_os_error(libc::EIO).into())
            },
            |delay| delays.push(delay.as_secs()),
        );
        assert!(is_eio(&result.unwrap_err()));
        assert_eq!(attempts.get(), 5);
        assert_eq!(opens.get(), 5);
        assert_eq!(delays, [1, 2, 4, 8]);
    }

    #[test]
    fn other_errors_fail_without_retry() {
        for error in [
            ParquetError::from(std::io::Error::from_raw_os_error(libc::ENOENT)),
            ParquetError::General("EIO mentioned in invalid data".into()),
        ] {
            let mut failure = Some(error);
            let mut reader = Some(());
            let result: ParquetResult<()> = read_group_with_retry(
                &mut reader,
                || panic!("non-EIO must not reopen"),
                |_| Err(failure.take().expect("one read attempt")),
                |_| panic!("non-EIO must not sleep"),
            );
            assert!(result.is_err());
        }
    }
}
