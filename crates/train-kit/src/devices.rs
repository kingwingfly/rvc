//! Guards on the device list a data-parallel run is given.

use anyhow::Result;

/// Reject a device list with repeats, *after* resolution.
///
/// Checking the `--device` strings is not enough: `auto,gpu:0` are different
/// spellings that name device 0 twice, and on WebGPU `auto`, `vulkan` and `mps`
/// all resolve to the default adapter. A repeat would silently double that
/// device's share of the work — and its memory — which looks like training
/// working.
///
/// Takes the resolved list by value and hands it back, so it reads as a step in
/// the resolution chain rather than as an assertion a caller can forget.
pub fn distinct<D: PartialEq + std::fmt::Debug>(devices: Vec<D>) -> Result<Vec<D>> {
    for (i, d) in devices.iter().enumerate() {
        anyhow::ensure!(
            !devices[..i].contains(d),
            "--device lists {d:?} more than once (different spellings can name one device)"
        );
    }
    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::distinct;

    #[test]
    fn a_repeated_device_is_rejected() {
        assert!(distinct(vec![0, 1, 2]).is_ok());
        let err = distinct(vec![0, 1, 0]).unwrap_err().to_string();
        assert!(err.contains("more than once"), "{err}");
    }
}
