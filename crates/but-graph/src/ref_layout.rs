//! The metadata-driven REF PLACEMENT table: which references sit on which commit, grouped
//! and ordered, one chain per metadata stack list.
//!
//! Authored by the builder's chain plan and stored on the [`CommitGraph`](crate::CommitGraph),
//! so the placement decisions survive the build. The chain-structure pass consumes the table
//! directly (a debug assert next to `chain_plan` proves it round-trips the plan), and the
//! rebase editor consumes the derived [`RefPositions`](crate::ref_layout::RefPositions).
//!
//! Remote-tracking references stay OUT of the table: they are disk-derived enrichment, not
//! placement decisions.

use std::collections::HashMap;

/// The ref placement table.
#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct RefLayout {
    /// The groups anchored on each commit, in chain-threading order.
    pub at_commit: HashMap<gix::ObjectId, Vec<LayoutGroup>>,
    /// One chain per metadata stack list, in metadata order.
    pub chains: Vec<Chain>,
    /// Commits kept anonymous (sorted): a shared base loses its build-time name so every
    /// chain's branches float above it as their own chain.
    pub anonymous_bases: Vec<gix::ObjectId>,
    /// The DERIVED editor-grade layout over the full ref universe (see [`RefPositions`]).
    /// `None` until the builder derives it from the authored segments.
    pub positions: Option<RefPositions>,
}

/// One metadata stack list's groups, in metadata order (top → bottom). Each anchor is
/// `(commit, index into RefLayout::at_commit[commit])` — the index keeps chains
/// apart when several chains anchor groups on the same commit.
#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct Chain {
    /// The chain's anchors in metadata order.
    pub anchors: Vec<(gix::ObjectId, usize)>,
}

/// One same-commit group of references anchored on a commit.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LayoutGroup {
    /// The member naming the anchor commit's segment, when the group names it at all.
    pub naming_ref: Option<NamingRef>,
    /// How the group's members land on the graph.
    pub placement: GroupPlacement,
}

/// The group member that NAMES the anchor commit's segment.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct NamingRef {
    /// The naming reference.
    pub name: gix::refs::FullName,
    /// The metadata-order override: this ref displaced a build-time name belonging to the
    /// group, whose remote link moves to its floated empty segment instead.
    pub clear_remote: bool,
}

/// How a group's members land relative to the commit the group anchors on.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum GroupPlacement {
    /// The group is outside the workspace or co-located with a managed merge commit — nothing
    /// is created, naming ref included. Kept so group ordinals stay aligned between plan and
    /// build.
    Skipped,
    /// Another chain owns the (non-integrated) commit: the members stay passive on it.
    Passive(Vec<gix::refs::FullName>),
    /// The non-naming members become empty segments spliced above the anchor.
    Splice {
        /// The spliced members, in metadata order.
        members: Vec<gix::refs::FullName>,
        /// The anchor commit is inside another chain, so the empties splice into it; otherwise
        /// the group anchors its own chain from the workspace (shared base or integrated
        /// anchor).
        into_owning_chain: bool,
    },
}

/// EVERY reference the workspace surfaces — chain names, empties, floats, remote and target
/// names, passive commit refs — with its resolved position over the commit graph, in segment
/// order. Derived from the build's authored segments; consumed by the rebase editor, which ingests it
/// 1:1 into its reference table.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RefPositions {
    /// The references in segment order — the order fixes the editor's reference table (and
    /// with it render sibling order).
    pub refs: Vec<PositionedRef>,
    /// The managed entrypoint commit and its resolved CHAIN slots, one per chain top→bottom —
    /// empty chains over one base yield duplicate entries the real commit does not have.
    /// `None` without a managed entrypoint commit.
    pub workspace_chain_slots: Option<(gix::ObjectId, Vec<gix::ObjectId>)>,
    /// Ordinals (into [`Self::refs`]) of the entrypoint's ref — the editor's HEAD checkouts.
    pub head_refs: Vec<usize>,
    /// Commits reachable from the entrypoint (sorted) — the editor's mutable commits.
    pub reachable_commits: Vec<gix::ObjectId>,
}

/// One reference of [`RefPositions`].
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PositionedRef {
    /// The reference name.
    pub name: gix::refs::FullName,
    /// Whether the entrypoint reaches this ref's segment — mutability before the editor's
    /// category gates (remote-category refs are never mutable).
    pub reachable: bool,
    /// Where the ref sits. `None` for unborn refs, which keep no stored position.
    pub position: Option<RefPosition>,
}

/// A reference's resolved position over the commit graph.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RefPosition {
    /// The commit the ref sits on.
    pub on: gix::ObjectId,
    /// The ordinal (into [`RefPositions::refs`]) of the next ref BELOW this one on the same
    /// commit run, when any.
    pub below: Option<usize>,
    /// The edges entering the ref from above, sorted: `(child commit, parent slot)` — the
    /// child reaches its parent slot's commit through this ref.
    pub entering: Vec<(gix::ObjectId, usize)>,
    /// Whether several edges converge right above the ref.
    pub ambiguous: bool,
}
