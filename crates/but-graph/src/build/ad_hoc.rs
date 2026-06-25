//! Ad-hoc/single-branch mode: decide which persisted GitButler-created branch orderings apply
//! to the checked-out ref. The structural rewrite happens on the segment data
//! (`segment_data::replay_ad_hoc`), and the final render installs the rewritten segments.

use anyhow::Context as _;
use but_core::RefMetadata;
use gix::reference::Category;

use crate::{
    Graph,
    walk::overlay::{OverlayMetadata, OverlayRepo},
};

impl Graph {
    /// In ad-hoc/single-branch mode, record the persisted GitButler-created branch ordering
    /// that shares the entrypoint's tip — the same-tip ref bundle to split into empty stack
    /// segments — in [`ad_hoc_branch_stack_orders`](Graph::ad_hoc_branch_stack_orders).
    pub(crate) fn ad_hoc_branch_stack_upgrades<T: RefMetadata>(
        &mut self,
        repo: &OverlayRepo<'_>,
        meta: &OverlayMetadata<'_, T>,
    ) -> anyhow::Result<()> {
        let Some(entrypoint_ref) = self.entrypoint_ref.clone() else {
            return Ok(());
        };
        if entrypoint_ref.category() != Some(Category::LocalBranch) {
            return Ok(());
        }
        if self.entrypoint.is_none() {
            return Ok(());
        }
        let Some(branch_order) = meta.branch_stack_order(entrypoint_ref.as_ref())? else {
            return Ok(());
        };
        let mut existing_ordered_refs = Vec::new();
        for branch in branch_order {
            if branch.category() != Some(Category::LocalBranch) {
                continue;
            }
            if repo
                .try_find_reference(branch.as_ref())
                .with_context(|| {
                    format!(
                        "failed to find ordered ad-hoc branch '{}'",
                        branch.shorten()
                    )
                })?
                .is_some()
            {
                existing_ordered_refs.push(branch);
            }
        }
        if existing_ordered_refs.len() < 2 {
            return Ok(());
        }

        let Some((bottom_ref, _)) = existing_ordered_refs.split_last() else {
            return Ok(());
        };
        let Some(mut bottom_reference) = repo
            .try_find_reference(bottom_ref.as_ref())
            .with_context(|| {
                format!(
                    "failed to find bottom ordered ad-hoc branch '{}'",
                    bottom_ref.shorten()
                )
            })?
        else {
            return Ok(());
        };
        let bottom_commit_id = bottom_reference
            .peel_to_id()
            .with_context(|| {
                format!(
                    "failed to peel bottom ordered ad-hoc branch '{}'",
                    bottom_ref.shorten()
                )
            })?
            .detach();

        let mut matching_refs = Vec::new();
        for branch in &existing_ordered_refs {
            let Some(mut reference) =
                repo.try_find_reference(branch.as_ref()).with_context(|| {
                    format!(
                        "failed to find ordered ad-hoc branch '{}'",
                        branch.shorten()
                    )
                })?
            else {
                continue;
            };
            if reference
                .peel_to_id()
                .with_context(|| {
                    format!(
                        "failed to peel ordered ad-hoc branch '{}'",
                        branch.shorten()
                    )
                })?
                .detach()
                == bottom_commit_id
            {
                matching_refs.push(branch.clone());
            }
        }
        if matching_refs.len() < 2 {
            return Ok(());
        }
        self.ad_hoc_branch_stack_orders.push(matching_refs);
        Ok(())
    }
}
