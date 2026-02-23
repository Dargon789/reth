//! Per-block post-execution handle for background post-execution artifact computation.
//!
//! This module provides [`PostExecHandle`], a block-scoped facade that spawns a single
//! event-driven background worker via [`Runtime::spawn_blocking_named`]. Receipts are
//! streamed incrementally during execution; after execution completes, a `Done` event
//! triggers receipt-root finalization and hashed-post-state computation. The same worker
//! also computes withdrawals root once at startup.
//!
//! Results are stored in shared [`OnceLock`] fields and accessed via [`OnceLock::wait`],
//! which blocks until the value is available.

use super::receipt_root_task::IndexedReceipt;
use alloy_eips::{eip4895::Withdrawal, Encodable2718};
use alloy_primitives::{Bloom, B256};
use crossbeam_channel::Sender as CrossbeamSender;
use reth_primitives_traits::{proofs::calculate_withdrawals_root, Receipt};
use reth_tasks::Runtime;
use reth_trie::HashedPostState;
use reth_trie_common::ordered_root::OrderedTrieRootEncodedBuilder;
use std::sync::{Arc, OnceLock};
use tracing::error;

/// Block-scoped handle for post-execution background tasks.
///
/// Created once per block via [`PostExecHandle::new`], which immediately spawns a single
/// background worker. During transaction execution, receipts are streamed via
/// [`push_receipt`](Self::push_receipt). After execution completes, call
/// [`finish`](Self::finish) to send the hashed-post-state closure and close the channel.
///
/// Results are stored in shared [`OnceLock`] fields and resolved lazily via
/// [`OnceLock::wait`].
pub struct PostExecHandle<R> {
    tx: Option<CrossbeamSender<PostExecEvent<R>>>,
    receipt_root_bloom: Arc<OnceLock<Option<(B256, Bloom)>>>,
    withdrawals_root: Arc<OnceLock<Option<B256>>>,
    hashed_post_state: Arc<OnceLock<Option<HashedPostState>>>,
}

impl<R> core::fmt::Debug for PostExecHandle<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PostExecHandle").finish()
    }
}

impl<R: Receipt + 'static> PostExecHandle<R> {
    /// Creates a new handle and immediately spawns the post-exec background worker.
    ///
    /// The worker begins waiting for events via the crossbeam channel and builds the
    /// receipt trie incrementally as receipts arrive.
    pub fn new(
        executor: &Runtime,
        receipts_len: usize,
        withdrawals: Option<Vec<Withdrawal>>,
    ) -> Self {
        let (tx, rx) = crossbeam_channel::unbounded();
        let receipt_root_bloom = Arc::new(OnceLock::new());
        let withdrawals_root = Arc::new(OnceLock::new());
        let hashed_post_state = Arc::new(OnceLock::new());
        let receipt_root_bloom_worker = receipt_root_bloom.clone();
        let withdrawals_root_worker = withdrawals_root.clone();
        let hashed_post_state_worker = hashed_post_state.clone();

        executor.spawn_blocking_named("post-exec", move || {
            run_post_exec_worker(
                rx,
                receipt_root_bloom_worker,
                withdrawals_root_worker,
                hashed_post_state_worker,
                receipts_len,
                withdrawals,
            );
        });

        Self { tx: Some(tx), receipt_root_bloom, withdrawals_root, hashed_post_state }
    }

    /// Streams one receipt to the background worker.
    #[inline]
    pub fn push_receipt(&self, index: usize, receipt: R) {
        if self.tx.as_ref().is_some_and(|tx| {
            tx.send(PostExecEvent::Receipt(IndexedReceipt::new(index, receipt))).is_err()
        }) {
            error!(
                target: "engine::tree::payload_processor",
                index,
                "post-exec worker dropped before receipt event",
            );
        }
    }

    /// Sends the `Done` event with the hashed-post-state closure and closes the channel.
    ///
    /// The background worker will finalize the receipt root, then invoke `f` to compute
    /// the hashed post state. Must be called after all receipts have been pushed.
    pub fn finish(&mut self, f: impl FnOnce() -> HashedPostState + Send + 'static) {
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(PostExecEvent::Done(Box::new(f)));
        }
    }

    /// Returns the computed receipt root and aggregated logs bloom.
    ///
    /// Blocks until the background worker completes receipt-root computation. Returns
    /// `None` if the receipt stream was incomplete (e.g., execution was aborted).
    pub fn receipt_root_bloom(&self) -> Option<(B256, Bloom)> {
        *self.receipt_root_bloom.wait()
    }

    /// Returns the computed withdrawals root.
    ///
    /// Blocks until the background worker completes withdrawals-root computation.
    pub fn withdrawals_root(&self) -> Option<B256> {
        *self.withdrawals_root.wait()
    }

    /// Returns a reference to the computed hashed post state.
    ///
    /// Blocks until the background task completes.
    ///
    /// # Panics
    ///
    /// Panics if the post-exec worker was aborted before computing the hashed post state.
    pub fn hashed_post_state(&self) -> &HashedPostState {
        self.hashed_post_state
            .wait()
            .as_ref()
            .expect("post-exec worker aborted before computing hashed post state")
    }

    /// Extracts a [`LazyHashedPostState`] wrapper from this handle.
    pub fn into_lazy_hashed_state(self) -> LazyHashedPostState {
        LazyHashedPostState { hashed_post_state: self.hashed_post_state }
    }
}

/// Event sent from the main execution thread to the post-exec background worker.
enum PostExecEvent<R> {
    /// A receipt produced during transaction execution.
    Receipt(IndexedReceipt<R>),
    /// Execution is complete; the closure computes the hashed post state.
    Done(Box<dyn FnOnce() -> HashedPostState + Send>),
}

/// Handle to a [`HashedPostState`] computed on a background thread.
///
/// Wraps the hashed-state result lock and provides a `LazyHandle`-compatible API so downstream
/// code that calls `.get()`, `.clone()`, and `.try_into_inner()` continues to work.
#[derive(Clone)]
pub struct LazyHashedPostState {
    hashed_post_state: Arc<OnceLock<Option<HashedPostState>>>,
}

impl core::fmt::Debug for LazyHashedPostState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut s = f.debug_struct("LazyHashedPostState");
        if let Some(value) = self.hashed_post_state.get() {
            s.field("value", &value.is_some());
        } else {
            s.field("value", &"<pending>");
        }
        s.finish()
    }
}

impl LazyHashedPostState {
    /// Blocks until the background worker completes and returns a reference to the result.
    pub fn get(&self) -> &HashedPostState {
        self.hashed_post_state
            .wait()
            .as_ref()
            .expect("post-exec worker aborted before computing hashed post state")
    }

    /// Consumes the handle and returns the inner value if this is the only reference.
    ///
    /// Returns `Err(self)` if other clones exist.
    /// Blocks if the background worker hasn't completed yet.
    pub fn try_into_inner(self) -> Result<HashedPostState, Self> {
        self.get();
        match Arc::try_unwrap(self.hashed_post_state) {
            Ok(inner) => Ok(inner
                .into_inner()
                .expect("value was just set by get()")
                .expect("post-exec worker aborted")),
            Err(arc) => Err(Self { hashed_post_state: arc }),
        }
    }
}

/// Runs the single post-exec background worker.
///
/// Receives events from the main execution thread: [`PostExecEvent::Receipt`] during
/// execution, then [`PostExecEvent::Done`] after execution completes. Incrementally builds
/// the receipt trie, finalizes it, then computes the hashed post state.
///
/// An RAII guard ensures all [`OnceLock`] fields are set on every exit path (including
/// panics), so [`OnceLock::wait`] never hangs.
fn run_post_exec_worker<R: Receipt>(
    rx: crossbeam_channel::Receiver<PostExecEvent<R>>,
    receipt_root_bloom: Arc<OnceLock<Option<(B256, Bloom)>>>,
    withdrawals_root: Arc<OnceLock<Option<B256>>>,
    hashed_post_state: Arc<OnceLock<Option<HashedPostState>>>,
    receipts_len: usize,
    withdrawals: Option<Vec<Withdrawal>>,
) {
    // RAII guard: sets any unset OnceLocks on drop (abort safety).
    // OnceLock::set is first-writer-wins, so successful sets are not overwritten.
    struct AbortGuard<'a> {
        receipt_root_bloom: &'a OnceLock<Option<(B256, Bloom)>>,
        withdrawals_root: &'a OnceLock<Option<B256>>,
        hashed_post_state: &'a OnceLock<Option<HashedPostState>>,
        disarmed: bool,
    }
    impl Drop for AbortGuard<'_> {
        fn drop(&mut self) {
            if !self.disarmed {
                let _ = self.receipt_root_bloom.set(None);
                let _ = self.withdrawals_root.set(None);
                let _ = self.hashed_post_state.set(None);
            }
        }
    }

    let mut guard = AbortGuard {
        receipt_root_bloom: &receipt_root_bloom,
        withdrawals_root: &withdrawals_root,
        hashed_post_state: &hashed_post_state,
        disarmed: false,
    };

    let _ = withdrawals_root.set(withdrawals.as_deref().map(calculate_withdrawals_root));

    let mut builder = OrderedTrieRootEncodedBuilder::new(receipts_len);
    let mut aggregated_bloom = Bloom::ZERO;
    let mut encode_buf = Vec::new();
    let mut received_count = 0usize;

    // Process events until Done or channel close.
    let done_f = loop {
        match rx.recv() {
            Ok(PostExecEvent::Receipt(indexed_receipt)) => {
                let receipt_with_bloom = indexed_receipt.receipt.with_bloom_ref();

                encode_buf.clear();
                receipt_with_bloom.encode_2718(&mut encode_buf);
                match builder.push(indexed_receipt.index, &encode_buf) {
                    Ok(()) => {
                        aggregated_bloom |= *receipt_with_bloom.bloom_ref();
                        received_count += 1;
                    }
                    Err(err) => {
                        error!(
                            target: "engine::tree::payload_processor",
                            index = indexed_receipt.index,
                            ?err,
                            "Post-exec worker received invalid receipt index, skipping"
                        );
                    }
                }
            }
            Ok(PostExecEvent::Done(f)) => break f,
            Err(_) => {
                // Channel closed before Done — execution was aborted.
                // Guard will set all OnceLocks to None.
                return;
            }
        }
    };

    // Finalize receipt root.
    match builder.finalize() {
        Ok(root) => {
            let _ = receipt_root_bloom.set(Some((root, aggregated_bloom)));
        }
        Err(_) => {
            error!(
                target: "engine::tree::payload_processor",
                expected = receipts_len,
                received = received_count,
                "Post-exec worker received incomplete receipts, execution likely aborted"
            );
            let _ = receipt_root_bloom.set(None);
        }
    }

    // Compute hashed post state.
    let hashed = done_f();
    let _ = hashed_post_state.set(Some(hashed));

    // All OnceLocks set successfully — disarm abort fallback.
    guard.disarmed = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_consensus::{proofs::calculate_receipt_root, TxReceipt};
    use alloy_eips::eip4895::Withdrawal;
    use alloy_primitives::{Address, Bytes, Log, B256};
    use reth_ethereum_primitives::{Receipt, TxType};

    fn test_runtime() -> Runtime {
        Runtime::test()
    }

    fn sample_receipts() -> Vec<Receipt> {
        vec![
            Receipt {
                tx_type: TxType::Legacy,
                cumulative_gas_used: 21_000,
                success: true,
                logs: vec![],
            },
            Receipt {
                tx_type: TxType::Eip1559,
                cumulative_gas_used: 42_000,
                success: true,
                logs: vec![Log {
                    address: Address::ZERO,
                    data: alloy_primitives::LogData::new_unchecked(vec![B256::ZERO], Bytes::new()),
                }],
            },
            Receipt {
                tx_type: TxType::Eip2930,
                cumulative_gas_used: 63_000,
                success: false,
                logs: vec![],
            },
        ]
    }

    fn sample_withdrawals() -> Vec<Withdrawal> {
        vec![Withdrawal { index: 1, validator_index: 2, address: Address::ZERO, amount: 3 }]
    }

    fn expected_root_bloom(receipts: &[Receipt]) -> (B256, Bloom) {
        let receipts_with_bloom: Vec<_> = receipts.iter().map(|r| r.with_bloom_ref()).collect();
        let root = calculate_receipt_root(&receipts_with_bloom);
        let bloom =
            receipts_with_bloom.iter().fold(Bloom::ZERO, |acc, receipt| acc | *receipt.bloom_ref());
        (root, bloom)
    }

    #[test]
    fn post_exec_handle_computes_receipt_root_and_bloom() {
        let rt = test_runtime();

        let receipts = sample_receipts();
        let (expected_root, expected_bloom) = expected_root_bloom(&receipts);

        let mut handle = PostExecHandle::<Receipt>::new(&rt, receipts.len(), None);
        for (index, receipt) in receipts.into_iter().enumerate() {
            handle.push_receipt(index, receipt);
        }
        handle.finish(HashedPostState::default);

        let (root, bloom) = handle.receipt_root_bloom().unwrap();
        assert_eq!(root, expected_root);
        assert_eq!(bloom, expected_bloom);
    }

    #[test]
    fn post_exec_handle_handles_out_of_order_receipts() {
        let rt = test_runtime();

        let receipts = sample_receipts();
        let (expected_root, expected_bloom) = expected_root_bloom(&receipts);

        let mut handle = PostExecHandle::<Receipt>::new(&rt, receipts.len(), None);
        for (index, receipt) in receipts.into_iter().enumerate().rev() {
            handle.push_receipt(index, receipt);
        }
        handle.finish(HashedPostState::default);

        let (root, bloom) = handle.receipt_root_bloom().unwrap();
        assert_eq!(root, expected_root);
        assert_eq!(bloom, expected_bloom);
    }

    #[test]
    fn post_exec_handle_ignores_invalid_index_for_bloom_aggregation() {
        let rt = test_runtime();

        let valid = Receipt::default();
        let invalid = Receipt {
            tx_type: TxType::Legacy,
            cumulative_gas_used: 21_000,
            success: true,
            logs: vec![Log {
                address: Address::ZERO,
                data: alloy_primitives::LogData::new_unchecked(vec![B256::ZERO], Bytes::new()),
            }],
        };

        let expected = expected_root_bloom(core::slice::from_ref(&valid));

        let mut handle = PostExecHandle::<Receipt>::new(&rt, 1, None);
        handle.push_receipt(0, valid);
        handle.push_receipt(999, invalid);
        handle.finish(HashedPostState::default);

        assert_eq!(handle.receipt_root_bloom(), Some(expected));
    }

    #[test]
    fn post_exec_handle_returns_none_for_incomplete_stream() {
        let rt = test_runtime();

        let mut handle = PostExecHandle::<Receipt>::new(&rt, 2, None);
        handle.push_receipt(0, Receipt::default());
        // Finish with only 1 of 2 receipts — root should be None.
        handle.finish(HashedPostState::default);

        assert!(handle.receipt_root_bloom().is_none());
    }

    #[test]
    fn post_exec_handle_with_hashed_post_state() {
        let rt = test_runtime();

        let mut handle = PostExecHandle::<Receipt>::new(&rt, 0, None);
        let expected = HashedPostState::default();
        handle.finish(HashedPostState::default);

        assert_eq!(handle.hashed_post_state(), &expected);
    }

    #[test]
    fn post_exec_handle_computes_withdrawals_root() {
        let rt = test_runtime();

        let withdrawals = sample_withdrawals();
        let expected = calculate_withdrawals_root(&withdrawals);

        let mut handle = PostExecHandle::<Receipt>::new(&rt, 0, Some(withdrawals));
        handle.finish(HashedPostState::default);

        assert_eq!(handle.withdrawals_root(), Some(expected));
    }

    #[test]
    fn post_exec_handle_parallel_blocks() {
        let rt = test_runtime();

        let receipts_a = sample_receipts();
        let (expected_root_a, expected_bloom_a) = expected_root_bloom(&receipts_a);

        let receipts_b = vec![Receipt::default(); 2];
        let (expected_root_b, expected_bloom_b) = expected_root_bloom(&receipts_b);

        let mut handle_a = PostExecHandle::<Receipt>::new(&rt, receipts_a.len(), None);
        let mut handle_b = PostExecHandle::<Receipt>::new(&rt, receipts_b.len(), None);

        for (index, receipt) in receipts_a.into_iter().enumerate() {
            handle_a.push_receipt(index, receipt);
        }
        for (index, receipt) in receipts_b.into_iter().enumerate() {
            handle_b.push_receipt(index, receipt);
        }

        handle_a.finish(HashedPostState::default);
        handle_b.finish(HashedPostState::default);

        let (root_a, bloom_a) = handle_a.receipt_root_bloom().unwrap();
        let (root_b, bloom_b) = handle_b.receipt_root_bloom().unwrap();

        assert_eq!(root_a, expected_root_a);
        assert_eq!(bloom_a, expected_bloom_a);
        assert_eq!(root_b, expected_root_b);
        assert_eq!(bloom_b, expected_bloom_b);
    }

    #[test]
    fn post_exec_handle_aborted_block_then_next_succeeds() {
        let rt = test_runtime();

        // First block: aborted (dropped without finishing all receipts)
        let handle = PostExecHandle::<Receipt>::new(&rt, 2, None);
        handle.push_receipt(0, Receipt::default());
        drop(handle);

        // Second block: succeeds
        let mut handle = PostExecHandle::<Receipt>::new(&rt, 1, None);
        handle.push_receipt(0, Receipt::default());
        handle.finish(HashedPostState::default);
        assert!(handle.receipt_root_bloom().is_some());
    }

    #[test]
    fn lazy_hashed_post_state_get_and_try_into_inner() {
        let rt = test_runtime();

        let mut handle = PostExecHandle::<Receipt>::new(&rt, 0, None);
        handle.finish(HashedPostState::default);

        let lazy = handle.into_lazy_hashed_state();
        assert_eq!(lazy.get(), &HashedPostState::default());

        // try_into_inner succeeds because into_lazy_hashed_state consumed the handle,
        // leaving only one Arc reference.
        let inner = lazy.try_into_inner().unwrap();
        assert_eq!(inner, HashedPostState::default());
    }

    #[test]
    fn lazy_hashed_post_state_clone_prevents_try_into_inner() {
        let rt = test_runtime();

        let mut handle = PostExecHandle::<Receipt>::new(&rt, 0, None);
        handle.finish(HashedPostState::default);

        let lazy = handle.into_lazy_hashed_state();
        let _clone = lazy.clone();

        // try_into_inner fails because there are multiple Arc references.
        let lazy = lazy.try_into_inner().unwrap_err();
        assert_eq!(lazy.get(), &HashedPostState::default());
    }
}
