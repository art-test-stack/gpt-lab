use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyMemoryView, PySlice, PyTuple};

mod parquet_source;
mod text_loader;

fn read_document(
    py: Python<'_>,
    document: &Bound<'_, PyAny>,
    tokens: &mut Vec<i64>,
    offset: usize,
    capacity: usize,
) -> PyResult<usize> {
    let buffer = PyBuffer::<i64>::get(document)?;
    if buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Document must be a contiguous int64 vector",
        ));
    }
    let length = buffer.item_count();
    if offset > length {
        return Err(PyValueError::new_err("Invalid pending document offset"));
    }
    let end = offset.saturating_add(capacity).min(length);
    tokens.resize(end - offset, 0);
    if offset == 0 && end == length {
        buffer.copy_to_slice(py, tokens)?;
    } else {
        let view = PyMemoryView::from(document)?;
        let slice = view.get_item(PySlice::new(py, offset as isize, end as isize, 1))?;
        PyBuffer::<i64>::get(&slice)?.copy_to_slice(py, tokens)?;
    }
    Ok(length)
}

fn select_document<E>(
    count: usize,
    space: usize,
    mut length_at: impl FnMut(usize) -> Result<usize, E>,
) -> Result<Option<usize>, E> {
    let mut best_fit = None;
    let mut shortest = None;
    for index in 0..count {
        let length = length_at(index)?;
        if length <= space && best_fit.is_none_or(|(_, best)| length > best) {
            best_fit = Some((index, length));
        }
        if shortest.is_none_or(|(_, best)| length < best) {
            shortest = Some((index, length));
        }
    }
    Ok(best_fit.or(shortest).map(|(index, _)| index))
}

#[pyfunction]
#[pyo3(signature = (out, lookahead, refill, document, document_ref, offset, bos))]
fn pack<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyAny>,
    lookahead: &Bound<'py, PyList>,
    refill: &Bound<'py, PyAny>,
    mut document: Bound<'py, PyAny>,
    mut document_ref: Bound<'py, PyAny>,
    mut offset: usize,
    bos: Option<i64>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, usize)> {
    let output = PyBuffer::<i64>::get(out)?;
    if output.dimensions() != 2
        || !output.is_c_contiguous()
        || output.readonly()
        || output.shape()[0] == 0
        || output.shape()[1] < 2
    {
        return Err(PyValueError::new_err(
            "Output must be a writable contiguous int64 batch",
        ));
    }
    let capacity = output.shape()[1];
    let mut values = vec![0; output.item_count()];
    let mut tokens = Vec::new();
    let mut slice_start = offset;
    let mut document_length = 0;
    if !document.is_none() {
        let continuation_bos =
            usize::from(bos.is_some()) * (output.shape()[0] - 1 + usize::from(offset > 0));
        document_length = read_document(
            py,
            &document,
            &mut tokens,
            offset,
            output.item_count() - continuation_bos,
        )?;
    }
    if (document.is_none() && offset != 0) || (!document.is_none() && offset >= document_length) {
        return Err(PyValueError::new_err("Invalid pending document offset"));
    }
    for row in 0..output.shape()[0] {
        let mut pos = 0;
        if offset > 0 {
            if let Some(token) = bos {
                values[row * capacity] = token;
                pos = 1;
            }
        }
        while pos < capacity {
            if document.is_none() {
                refill.call0()?;
                let index = select_document(lookahead.len(), capacity - pos, |index| {
                    lookahead
                        .get_item(index)?
                        .cast::<PyTuple>()?
                        .get_item(0)?
                        .len()
                })?
                .ok_or_else(|| PyStopIteration::new_err(()))?;
                let item = lookahead.get_item(index)?;
                let pair = item.cast::<PyTuple>()?;
                document = pair.get_item(0)?;
                document_ref = pair.get_item(1)?;
                slice_start = 0;
                let remaining_rows = output.shape()[0] - row - 1;
                let available =
                    capacity - pos + remaining_rows * (capacity - usize::from(bos.is_some()));
                document_length = read_document(py, &document, &mut tokens, 0, available)?;
                lookahead.del_item(index)?;
            }
            let take = (document_length - offset).min(capacity - pos);
            let start = row * capacity + pos;
            let token_start = offset - slice_start;
            values[start..start + take].copy_from_slice(&tokens[token_start..token_start + take]);
            pos += take;
            offset += take;
            if offset == document_length {
                document = py.None().into_bound(py);
                document_ref = py.None().into_bound(py);
                offset = 0;
            }
        }
    }
    output.copy_from_slice(py, &values)?;
    Ok((document, document_ref, offset))
}

#[pymodule]
fn gpt_lab_rust_packer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(pack, module)?)?;
    module.add_class::<text_loader::TextLoader>()?;
    Ok(())
}
