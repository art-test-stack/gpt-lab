use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;
use parquet::record::Field;
use parquet::schema::types::Type;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

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

    fn open(&self, index: usize) -> Result<SerializedFileReader<File>, String> {
        let path = &self.shards[index].1;
        let file =
            File::open(path).map_err(|error| format!("Cannot open {}: {error}", path.display()))?;
        SerializedFileReader::new(file)
            .map_err(|error| format!("Cannot read {}: {error}", path.display()))
    }

    fn read_group(
        &self,
        reader: &SerializedFileReader<File>,
        group: usize,
    ) -> Result<Vec<String>, String> {
        if group >= reader.num_row_groups() {
            return Err(format!(
                "Data checkpoint row group {group} is outside the shard"
            ));
        }
        let schema = reader.metadata().file_metadata().schema();
        let field = schema
            .get_fields()
            .iter()
            .find(|field| field.name() == self.column)
            .ok_or_else(|| format!("Missing text column {}", self.column))?;
        let projection = Type::group_type_builder(schema.name())
            .with_fields(vec![field.clone()])
            .build()
            .map_err(|error| error.to_string())?;
        let group_reader = reader
            .get_row_group(group)
            .map_err(|error| error.to_string())?;
        let rows = group_reader
            .get_row_iter(Some(projection))
            .map_err(|error| error.to_string())?;
        rows.map(|row| {
            let row = row.map_err(|error| error.to_string())?;
            match row.into_columns().pop() {
                Some((_, Field::Str(text))) => Ok(text),
                _ => Err(format!(
                    "Column {} must contain non-null UTF-8 text",
                    self.column
                )),
            }
        })
        .collect()
    }

    pub fn next_chunk(&mut self, limit: usize) -> Result<Option<(Vec<String>, Cursor)>, String> {
        if limit == 0 {
            return Err("Shard iteration batch size must be positive".into());
        }
        while !self.exhausted {
            if self.reader.is_none() {
                self.reader = Some(self.open(self.cursor.shard_idx)?);
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
                    let texts = self.read_group(reader, self.cursor.row_group_idx)?;
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
            groups.insert((shard, group), self.read_group(&self.open(index)?, group)?);
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
