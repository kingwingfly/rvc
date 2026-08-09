//! What a stage is given: which files to read, and what to call what it writes.
//!
//! Shared by every stage rather than owned by one, so a directory expanded for
//! `clip` and a directory expanded for `denoise` cannot come out different.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use audio_kit::{DecodeOptions, decode_paths};
use futures::StreamExt;

/// Audio extensions a directory is expanded into (case-insensitive).
pub const AUDIO_EXTS: &[&str] = &["mp3", "wav", "flac", "m4a", "ogg", "opus", "aac", "wma"];

/// One file to process, and the base name its output is written under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputFile {
    /// The file to read.
    pub path: PathBuf,
    /// Output base name, unique across the batch (see [`plan`]).
    pub base: String,
}

/// Resolve the paths a user named into the batch to process.
///
/// Files are kept as given; directories are walked **recursively** for audio
/// files by extension. The result is sorted and de-duplicated, so a run is
/// reproducible no matter what order the filesystem hands entries back.
///
/// Two files can share a stem (`a/take.mp3` and `b/take.mp3`) and would then
/// write over each other's output, so the *second* and later occurrences get
/// `__1`, `__2` … appended to their base. Silently losing half a corpus to a
/// name collision is the failure this prevents.
///
/// **`output_dir` must already exist** for the skip to work: the guard
/// canonicalises it, and a path that is not there yet canonicalises to nothing
/// to compare against. Callers create the directory first, which is also what
/// makes a re-run refuse to re-ingest its own output.
pub fn plan(inputs: &[PathBuf], output_dir: &Path) -> Result<Vec<InputFile>> {
    let skip = std::fs::canonicalize(output_dir).ok();
    let mut files = Vec::new();
    for path in inputs {
        if path.is_dir() {
            collect_dir(path, skip.as_deref(), &mut files)?;
        } else {
            files.push(path.clone());
        }
    }
    files.sort();
    files.dedup();

    let mut used: HashMap<String, usize> = HashMap::new();
    Ok(files
        .into_iter()
        .map(|path| {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("clip")
                .to_string();
            let seen = used.entry(stem.clone()).or_insert(0);
            let base = if *seen == 0 {
                stem
            } else {
                format!("{stem}__{seen}")
            };
            *seen += 1;
            InputFile { path, base }
        })
        .collect())
}

/// Recursively collect audio files under `dir`, skipping the output dir.
fn collect_dir(dir: &Path, skip: Option<&Path>, out: &mut Vec<PathBuf>) -> Result<()> {
    if let (Some(skip), Ok(here)) = (skip, std::fs::canonicalize(dir)) {
        if here == skip {
            return Ok(());
        }
    }
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("reading directory {}", dir.display()))?;
    for entry in entries {
        let p = entry?.path();
        if p.is_dir() {
            collect_dir(&p, skip, out)?;
        } else if p.is_file() && has_audio_ext(&p) {
            out.push(p);
        }
    }
    Ok(())
}

/// Whether `path`'s extension is a recognised audio format (case-insensitive).
fn has_audio_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let e = e.to_ascii_lowercase();
            AUDIO_EXTS.contains(&e.as_str())
        })
        .unwrap_or(false)
}

/// Decode a file to mono `f32` at `sr`, draining the whole stream into a `Vec`.
///
/// Whole-file rather than streaming because every stage here needs the length
/// before it can decide anything: the slicer balances a split across a voiced
/// run, and a de-hiss report is a fraction of a duration.
pub async fn decode_mono(input: &Path, sr: u32) -> Result<Vec<f32>> {
    let decode = decode_paths(vec![input.to_path_buf()], DecodeOptions::new(sr));
    let mut decode = std::pin::pin!(decode);
    let mut out = Vec::new();
    while let Some(chunk) = decode.next().await {
        out.extend(chunk.with_context(|| format!("decoding {}", input.display()))?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory of this test's own, named after the process so parallel
    /// worktrees cannot collide in `/tmp` — the same hazard the shared cargo
    /// target directory has.
    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("preprocess-core-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent");
        }
        std::fs::write(path, b"").expect("touch");
    }

    #[test]
    fn audio_extensions_are_case_insensitive() {
        assert!(has_audio_ext(Path::new("a.wav")));
        assert!(has_audio_ext(Path::new("a.WAV")));
        assert!(has_audio_ext(Path::new("a.Mp3")));
        assert!(!has_audio_ext(Path::new("a.txt")));
        // A transcript sitting beside its clip is the common case, and a
        // corpus directory is full of them.
        assert!(!has_audio_ext(Path::new("a")));
        assert!(!has_audio_ext(Path::new("a.wav.bak")));
    }

    #[test]
    fn directories_expand_recursively_sorted_and_deduped() {
        let dir = scratch("expand");
        touch(&dir.join("b.wav"));
        touch(&dir.join("a.mp3"));
        touch(&dir.join("notes.txt"));
        touch(&dir.join("nested/c.flac"));
        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");

        // The same directory twice, plus one file named directly: the file is
        // already in the walk, so it must not appear twice.
        let files = plan(
            &[dir.clone(), dir.clone(), dir.join("a.mp3")],
            &out,
        )
        .expect("plan");
        let paths: Vec<_> = files.iter().map(|f| f.path.clone()).collect();
        assert_eq!(
            paths,
            vec![dir.join("a.mp3"), dir.join("b.wav"), dir.join("nested/c.flac")]
        );
    }

    #[test]
    fn the_output_directory_is_never_re_ingested() {
        let dir = scratch("skip-output");
        touch(&dir.join("a.wav"));
        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        // A previous run's clips, sitting inside the input tree.
        touch(&out.join("a_000.wav"));

        let files = plan(&[dir.clone()], &out).expect("plan");
        assert_eq!(files.len(), 1, "the output dir's own clips were re-ingested");
        assert_eq!(files[0].path, dir.join("a.wav"));

        // The skip canonicalises `output_dir`, so it can only fire once the
        // directory exists. Callers create it first; pinned here because
        // nothing else would notice the ordering dependency.
        let missing = dir.join("not-yet");
        let files = plan(&[dir.clone()], &missing).expect("plan");
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn files_sharing_a_stem_get_distinct_output_bases() {
        let dir = scratch("stems");
        touch(&dir.join("a/take.wav"));
        touch(&dir.join("b/take.wav"));
        touch(&dir.join("c/take.wav"));
        touch(&dir.join("a/other.wav"));
        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");

        let bases: Vec<_> = plan(&[dir.clone()], &out)
            .expect("plan")
            .into_iter()
            .map(|f| f.base)
            .collect();
        // Sorted by path, so `a/other`, `a/take`, `b/take`, `c/take`.
        assert_eq!(bases, vec!["other", "take", "take__1", "take__2"]);
    }
}
