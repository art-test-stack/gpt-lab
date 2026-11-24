use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use tiktoken::CoreBPE;

use crate::parquet_source::{Cursor, CursorTuple, DocumentRef, TextSource};

type TokenizerConfig = (HashMap<Vec<u8>, u32>, String, HashMap<String, u32>, u32);
type State = (CursorTuple, Vec<DocumentRef>, Option<DocumentRef>, usize);
type Document = (Vec<u32>, Option<DocumentRef>);

enum FillError {
    Exhausted,
    Source(String),
}

#[pyclass(module = "gpt_lab_rust_packer")]
pub struct TextLoader {
    source: TextSource,
    tokenizer: CoreBPE,
    pool: rayon::ThreadPool,
    bos: u32,
    batch_size: usize,
    capacity: usize,
    encoding_batch_size: usize,
    training: bool,
    cursor: Cursor,
    encoded: VecDeque<(Vec<u32>, Cursor)>,
    lookahead: Vec<Document>,
    document: Option<Document>,
    offset: usize,
    output: Vec<i64>,
    failed: bool,
    exhausted: bool,
}

impl TextLoader {
    fn encode(&self, texts: &[String]) -> Vec<Vec<u32>> {
        self.pool.install(|| {
            texts
                .par_iter()
                .map(|text| {
                    let mut tokens = self.tokenizer.encode_ordinary(text);
                    tokens.insert(0, self.bos);
                    tokens
                })
                .collect()
        })
    }

    fn refill(&mut self) -> Result<(), FillError> {
        while self.lookahead.len() < 16 {
            if self.encoded.is_empty() {
                let Some((texts, cursor)) = self
                    .source
                    .next_chunk(self.encoding_batch_size)
                    .map_err(FillError::Source)?
                else {
                    break;
                };
                let first_offset = cursor.offset_in_row_group - texts.len() + 1;
                for (index, tokens) in self.encode(&texts).into_iter().enumerate() {
                    let mut document_cursor = cursor;
                    document_cursor.offset_in_row_group = first_offset + index;
                    self.encoded.push_back((tokens, document_cursor));
                }
            }
            let Some((_, cursor)) = self.encoded.front() else {
                continue;
            };
            if !self.lookahead.is_empty() && cursor.epoch != self.cursor.epoch {
                break;
            }
            let (tokens, cursor) = self.encoded.pop_front().unwrap();
            let reference = if self.training {
                Some((
                    cursor
                        .global_shard_idx
                        .ok_or_else(|| FillError::Source("Missing physical shard ID".into()))?,
                    cursor.row_group_idx,
                    cursor.offset_in_row_group - 1,
                ))
            } else {
                None
            };
            self.lookahead.push((tokens, reference));
            self.cursor = cursor;
        }
        Ok(())
    }

    fn pack(&mut self) -> Result<(), FillError> {
        for row in 0..self.batch_size {
            let mut position = 0;
            if self.offset > 0 {
                self.output[row * self.capacity] = i64::from(self.bos);
                position = 1;
            }
            while position < self.capacity {
                if self.document.is_none() {
                    self.refill()?;
                    let best = crate::select_document(
                        self.lookahead.len(),
                        self.capacity - position,
                        |index| {
                            Ok::<usize, std::convert::Infallible>(self.lookahead[index].0.len())
                        },
                    )
                    .unwrap()
                    .ok_or(FillError::Exhausted)?;
                    self.document = Some(self.lookahead.remove(best));
                }
                let tokens = &self.document.as_ref().unwrap().0;
                let take = (tokens.len() - self.offset).min(self.capacity - position);
                for (destination, token) in self.output
                    [row * self.capacity + position..row * self.capacity + position + take]
                    .iter_mut()
                    .zip(&tokens[self.offset..self.offset + take])
                {
                    *destination = i64::from(*token);
                }
                position += take;
                self.offset += take;
                if self.offset == tokens.len() {
                    self.document = None;
                    self.offset = 0;
                }
            }
        }
        Ok(())
    }

    fn state(&self) -> State {
        (
            self.cursor.into(),
            self.lookahead
                .iter()
                .filter_map(|(_, reference)| *reference)
                .collect(),
            self.document.as_ref().and_then(|(_, reference)| *reference),
            self.offset,
        )
    }
}

#[pymethods]
impl TextLoader {
    #[new]
    #[pyo3(signature = (shards, column, training, rank, world_size, batch_shape, tokenizer_threads,
                       tokenizer_batch_size, tokenizer, resume=None))]
    fn new(
        py: Python<'_>,
        shards: Vec<(usize, PathBuf)>,
        column: String,
        training: bool,
        rank: usize,
        world_size: usize,
        batch_shape: (usize, usize),
        tokenizer_threads: usize,
        tokenizer_batch_size: usize,
        tokenizer: TokenizerConfig,
        resume: Option<State>,
    ) -> PyResult<Self> {
        let (batch_size, capacity) = batch_shape;
        let elements = batch_size
            .checked_mul(capacity)
            .ok_or_else(|| PyValueError::new_err("Batch shape overflow"))?;
        if batch_size == 0 || capacity < 2 || tokenizer_threads == 0 || tokenizer_batch_size == 0 {
            return Err(PyValueError::new_err(
                "Batch shape, tokenizer threads and batch size must be positive",
            ));
        }
        py.detach(move || {
            let (merge_ranks, regex, special_tokens, bos) = tokenizer;
            if !special_tokens.values().any(|id| *id == bos) {
                return Err(PyValueError::new_err(
                    "BOS must be a configured special token",
                ));
            }
            let tokenizer = CoreBPE::new::<_, _, Vec<(String, (u32, u32))>>(
                merge_ranks,
                special_tokens,
                &regex,
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(tokenizer_threads)
                .build()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let source_cursor = resume
                .as_ref()
                .map(|(cursor, _, _, _)| Cursor::from(*cursor));
            let source = TextSource::new(shards, column, training, rank, world_size, source_cursor)
                .map_err(PyValueError::new_err)?;
            let mut loader = Self {
                source,
                tokenizer,
                pool,
                bos,
                batch_size,
                capacity,
                encoding_batch_size: tokenizer_batch_size,
                training,
                cursor: source_cursor.unwrap_or_default(),
                encoded: VecDeque::new(),
                lookahead: Vec::new(),
                document: None,
                offset: 0,
                output: vec![0; elements],
                failed: false,
                exhausted: false,
            };
            if let Some((_, lookahead_refs, pending, offset)) = resume {
                let mut refs = lookahead_refs.clone();
                refs.extend(pending);
                if lookahead_refs.len() >= 16
                    || refs.iter().collect::<HashSet<_>>().len() != refs.len()
                    || (pending.is_none() && offset != 0)
                {
                    return Err(PyValueError::new_err(
                        "Invalid packing buffer in data checkpoint",
                    ));
                }
                let texts = loader
                    .source
                    .read_documents(&refs)
                    .map_err(PyValueError::new_err)?;
                if texts.len() != refs.len() {
                    return Err(PyValueError::new_err("Missing checkpoint document"));
                }
                let mut documents = loader.encode(&texts).into_iter();
                for reference in lookahead_refs {
                    loader
                        .lookahead
                        .push((documents.next().unwrap(), Some(reference)));
                }
                if let Some(reference) = pending {
                    let tokens = documents.next().unwrap();
                    if offset == 0 || offset >= tokens.len() {
                        return Err(PyValueError::new_err("Invalid pending document offset"));
                    }
                    loader.document = Some((tokens, Some(reference)));
                    loader.offset = offset;
                }
            }
            Ok(loader)
        })
    }

    fn fill(&mut self, py: Python<'_>, out: &Bound<'_, PyAny>) -> PyResult<State> {
        if self.failed {
            return Err(PyRuntimeError::new_err(
                "Discard the failed iterator and restore the last committed DataLoaderState",
            ));
        }
        if self.exhausted {
            return Err(PyStopIteration::new_err(()));
        }
        let output = PyBuffer::<i64>::get(out)?;
        if output.dimensions() != 2
            || !output.is_c_contiguous()
            || output.readonly()
            || output.shape() != [self.batch_size, self.capacity]
        {
            return Err(PyValueError::new_err(
                "Output must be a writable contiguous int64 batch with the configured shape",
            ));
        }
        self.failed = true;
        match py.detach(|| self.pack()) {
            Ok(()) => (),
            Err(FillError::Exhausted) => {
                self.failed = false;
                self.exhausted = true;
                return Err(PyStopIteration::new_err(()));
            }
            Err(FillError::Source(error)) => {
                return Err(PyRuntimeError::new_err(error));
            }
        }
        output.copy_from_slice(py, &self.output)?;
        self.failed = false;
        Ok(self.state())
    }
}
