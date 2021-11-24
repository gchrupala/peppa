Each `JSON` file contains the following information:

- transcript - the text to align
- speaker - character speaking, if available (annotated only for a subset of cases, where a line is spoken by a single character)
- episode_filepath: which episode file this clip was extracted from
- episode_metadata_path: which metadata file the subtitles were taken from
- episode_title: title of the episode
- clipStart: offset in seconds where the clip starts in the original video file
- clipEnd: offset in seconds where the clip ends in the original video file
- partIndex, clipIndex: these refer to the sequential index of the narration/dialog section and the clip.
- words: aligned words (output of the  https://github.com/lowerquality/gentle aligned

Each word consists of:
 
 - case: {success, not-in-found-in-audio, ...}
 - alignedWord: the word form
 - start: beginning position in the clips in second (in order to find the position in the original video file this needs to be added to the `clipStart` value above)
 - end: end position in the clip in seconds (ditto)
 - phones: list of phoneme alignments
 
 Each phoneme has the following information:
 
 - phone: ARPA symbol for the phoneme, followed by the underscore and a tag indicating phoneme position. ARPA symbols are mapped to IPA in [../../../pig/ipa.py].
 - duration: duration in seconds.
 
