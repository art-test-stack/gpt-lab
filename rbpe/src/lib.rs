use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyIterator, PyList, PyTuple};
use memmap2::MmapOptions;
use std::fs::File;


/// Streaming BPE trainer placeholder
#[pyfunction]
fn stream_bpe<'py>(
    py: Python<'py>,
    iterator: &PyAny,
) -> PyResult<(PyObject, PyObject)> {

    // Convert to Python iterator
    let iter = PyIterator::from_object(iterator)?;

    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut vocab: Vec<Vec<u8>> = Vec::new();

    for item in iter {
        let item = item?;
        let list: &PyList = item.downcast()?;

        for element in list {
            let token: String = element.extract()?;
            vocab.push(token.into_bytes());
        }
    }

    // Dummy merge pair
    merges.push((b"ab".to_vec(), b"cd".to_vec()));

    // ---- Build Python objects ----

    // merges → List[Tuple[bytes, bytes]]
    let py_merges = PyList::new(
        py,
        merges.iter().map(|(a, b)| {
            PyTuple::new(py, &[PyBytes::new(py, a), PyBytes::new(py, b)])
        }),
    );

    // vocab → List[bytes]
    let py_vocab = PyList::new(
        py,
        vocab.iter().map(|v| PyBytes::new(py, v)),
    );

    Ok((py_merges.into(), py_vocab.into()))
}


/// mmap length test function
#[pyfunction]
fn mmap_len(path: String) -> PyResult<usize> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(mmap.len())
}


/// Module definition
#[pymodule]
fn streaming_bpe(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stream_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(mmap_len, m)?)?;
    Ok(())
}
