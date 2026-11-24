use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

fn document_buffer(document: &Bound<'_, PyAny>) -> PyResult<PyBuffer<i64>> {
    let buffer = PyBuffer::<i64>::get_bound(document)?;
    if buffer.dimensions() != 1 || !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "Document must be a contiguous int64 vector",
        ));
    }
    Ok(buffer)
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
    let output = PyBuffer::<i64>::get_bound(out)?;
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
    let values = output
        .as_mut_slice(py)
        .ok_or_else(|| PyValueError::new_err("Invalid output buffer"))?;
    let mut source = if document.is_none() {
        None
    } else {
        Some(document_buffer(&document)?)
    };
    if (source.is_none() && offset != 0)
        || source
            .as_ref()
            .is_some_and(|buffer| offset >= buffer.item_count())
    {
        return Err(PyValueError::new_err("Invalid pending document offset"));
    }
    for row in 0..output.shape()[0] {
        let mut pos = 0;
        if offset > 0 {
            if let Some(token) = bos {
                values[row * capacity].set(token);
                pos = 1;
            }
        }
        while pos < capacity {
            if source.is_none() {
                refill.call0()?;
                let mut best_fit = None;
                let mut shortest = None;
                for (index, item) in lookahead.iter().enumerate() {
                    let item = item.downcast::<PyTuple>()?;
                    let length = item.get_item(0)?.len()?;
                    if length <= capacity - pos && best_fit.is_none_or(|(_, best)| length > best) {
                        best_fit = Some((index, length));
                    }
                    if shortest.is_none_or(|(_, best)| length < best) {
                        shortest = Some((index, length));
                    }
                }
                let index = best_fit
                    .or(shortest)
                    .ok_or_else(|| PyStopIteration::new_err(()))?
                    .0;
                let item = lookahead.get_item(index)?;
                let pair = item.downcast::<PyTuple>()?;
                document = pair.get_item(0)?;
                document_ref = pair.get_item(1)?;
                source = Some(document_buffer(&document)?);
                lookahead.del_item(index)?;
            }
            let buffer = source.as_ref().unwrap();
            let tokens = buffer
                .as_slice(py)
                .ok_or_else(|| PyValueError::new_err("Invalid document buffer"))?;
            let take = (tokens.len() - offset).min(capacity - pos);
            for index in 0..take {
                values[row * capacity + pos + index].set(tokens[offset + index].get());
            }
            pos += take;
            offset += take;
            if offset == tokens.len() {
                source = None;
                document = py.None().into_bound(py);
                document_ref = py.None().into_bound(py);
                offset = 0;
            }
        }
    }
    Ok((document, document_ref, offset))
}

#[pymodule]
fn gpt_lab_rust_packer(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(pack, module)?)?;
    Ok(())
}
