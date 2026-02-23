//! Receipt type with transaction index for background receipt root computation.

/// Receipt with index, ready to be sent to the background task for encoding and trie building.
#[derive(Debug, Clone)]
pub struct IndexedReceipt<R> {
    /// The transaction index within the block.
    pub index: usize,
    /// The receipt.
    pub receipt: R,
}

impl<R> IndexedReceipt<R> {
    /// Creates a new indexed receipt.
    #[inline]
    pub const fn new(index: usize, receipt: R) -> Self {
        Self { index, receipt }
    }
}
