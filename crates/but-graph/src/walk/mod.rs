use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context as _, bail, ensure};
use but_core::{
    RefMetadata, extract_remote_name_and_short_name,
    ref_metadata::{self, ProjectMeta},
};
use gix::prelude::{ObjectIdExt, ReferenceExt};
use tracing::instrument;

use crate::{CommitFlags, Graph, SegmentMetadata};

pub(crate) mod utils;
use utils::*;

pub(crate) mod types;
use types::{Goals, Limit};

use crate::walk::overlay::{OverlayMetadata, OverlayRepo};

mod remotes;

pub(crate) mod overlay;
pub(crate) mod walker;

pub(crate) type Entrypoint = Option<(gix::ObjectId, Option<gix::refs::FullName>)>;

/// A resolved commit that seeds graph traversal without requiring it to be
/// discoverable through repository refs or workspace metadata.
///
/// ## Traversal invariants
///
/// The traversal will build a segment graph, where Segments follow specific rules.
/// We differentiate between [seed segments](crate::Segment), segments created from [Seed]s, (*TS*) and
/// ancestor segments (*AS*), which are ancestors of *TS* and connected to them by outgoing
/// connections.
///
/// - Virtual segments (*VS*) are minted by chain materialization to represent refs
///   which are described in [but_core::ref_metadata::Workspace]. They are [named](crate::Segment::ref_name())
///   and always empty, and ordinary virtual segments have *exactly one*
///   outgoing connection that lets `Graph::resolve_to_unambiguously_pointed_to_commit()`
///   find the commit named by the ref. The commit is owned by another segment, sometimes
///   because another segment was prioritized when multiple refs point to the same commit.
/// - The virtual workspace seed segment is a special kind of *VS*, which may have one or more
///   outgoing connections, pointing to one or more *VS* or *AS*. As such, such Segments cannot
///   unambiguously determine the commit their [Self::ref_name] points to as multiple paths can
///   be followed, yielding multiple commits.
///   Note that ordinary workspace seed segments may also exist as *TS*, which do own a commit,
///   which *typically* is the workspace commit.
/// - Forks and joins of the underlying
///   commit graph are represented by segments. This allows traversals or
///   graph computations, like merge-bases, to work the same as on the commit-graph, but
///   possibly with fewer jumps as segments may contain more than one commit,
///   allowing to skip over uninteresting commits naturally.
/// - The built graph may not fully represent the commit-graph
///   due to the creation of *VS*. What makes a *VS* virtual is not the ref itself,
///   but that its relationship to other segments is not represented by the Git
///   commit-graph or by Git refs: to Git, these are refs pointing to the same commit,
///   while GitButler sees one or more stacks of branches with specific ordering.
/// - *TS* with [Self::ref_name] set will return that as [crate::Segment::ref_name()].
/// - *TS* that contain [Self::id] contain it as first commit
/// - *TS* that don't contain [Self::id] are empty and can find their commit by following
///   their only outgoing connection until a non-empty commit is found which contains
///   [Self::id] as *first* commit!
/// - *TS* or *AS* with *more than one* outgoing connection have *at least one* commit.
#[derive(Debug, Clone)]
pub struct Seed {
    /// The commit id to start walking from.
    pub id: gix::ObjectId,
    /// The ref name to assign to the seed segment, if it should be named.
    pub ref_name: Option<gix::refs::FullName>,
    /// How this seed participates in traversal.
    pub role: SeedRole,
    /// Metadata to attach to the initial segment.
    pub metadata: Option<SegmentMetadata>,
    /// Whether this seed is the user-facing traversal entrypoint.
    ///
    /// There may only be *one such seed*.
    /// Other seeds try to connect to any commit reachable from this one.
    pub is_entrypoint: bool,
    /// Whether the entrypoint segment should remain anonymous even if refs
    /// point at the same commit.
    pub is_detached: bool,
}

/// Lifecycle
impl Seed {
    /// A traversal seed with default reachable semantics.
    ///
    /// This is the smallest seed description: it starts at `id`, is unnamed, is
    /// not the entrypoint, carries no metadata, and queues after existing
    /// initial work.
    pub fn new(id: gix::ObjectId) -> Self {
        Seed {
            id,
            ref_name: None,
            role: SeedRole::default(),
            metadata: None,
            is_entrypoint: false,
            is_detached: false,
        }
    }

    /// A normal named or unnamed traversal entrypoint.
    ///
    /// `id` is the commit where graph traversal starts.
    /// `ref_name` names the entrypoint segment when the caller has a stable ref
    /// for it.
    pub fn entrypoint(id: gix::ObjectId, ref_name: Option<gix::refs::FullName>) -> Self {
        Seed::new(id).with_ref_name(ref_name).with_entrypoint()
    }

    /// An entrypoint whose segment should remain detached even if refs point to
    /// its commit.
    ///
    /// `id` is the commit where graph traversal starts.
    pub fn detached_entrypoint(id: gix::ObjectId) -> Self {
        Seed::new(id).with_detached_entrypoint()
    }

    /// A non-remote seed that should be included in the traversal.
    ///
    /// `id` is the commit to include as another non-remote traversal root.
    /// `ref_name` names the seed segment when the caller has a stable ref for it.
    pub fn reachable(id: gix::ObjectId, ref_name: Option<gix::refs::FullName>) -> Self {
        Seed::new(id).with_ref_name(ref_name)
    }

    /// A target/integration seed that bounds or extends traversal context.
    /// It represents part of the graph that [`Self::reachable()`] parts want to integrate with.
    ///
    /// `id` is the commit to treat as integrated history.
    /// `ref_name` names the target segment when the caller has a stable ref for
    /// it.
    pub fn integrated(id: gix::ObjectId, ref_name: Option<gix::refs::FullName>) -> Self {
        Seed::new(id)
            .with_ref_name(ref_name)
            .with_role(SeedRole::TargetRemote)
    }
}

/// Builder
impl Seed {
    /// Set the ref name used to enforce the name this seed segment.
    pub fn with_ref_name(mut self, ref_name: Option<gix::refs::FullName>) -> Self {
        self.ref_name = ref_name;
        self
    }

    /// Set the traversal role for this seed.
    pub fn with_role(mut self, role: SeedRole) -> Self {
        self.role = role;
        self
    }

    /// Set whether this seed is the traversal entrypoint.
    pub(crate) fn with_is_entrypoint(mut self, is_entrypoint: bool) -> Self {
        self.is_entrypoint = is_entrypoint;
        self
    }

    /// Set whether this seed should use detached entrypoint presentation, which makes it anonymous even
    /// if it could receive a name/unambiguous ref otherwise.
    pub fn with_is_detached(mut self, is_detached: bool) -> Self {
        self.is_detached = is_detached;
        self
    }

    /// Mark this seed as the traversal entrypoint.
    pub fn with_entrypoint(self) -> Self {
        self.with_is_entrypoint(true)
    }

    /// Mark this entrypoint as detached for segment presentation.
    pub(crate) fn with_detached_entrypoint(mut self) -> Self {
        self = self.with_is_entrypoint(true).with_is_detached(true);
        self
    }

    /// Attach metadata to the initial segment created for this seed.
    pub fn with_metadata(mut self, metadata: SegmentMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Utilities
impl Seed {
    /// Return whether this seed is anonymous integrated target context.
    ///
    /// Named target remotes can represent refs that need their own segment and
    /// target/local sibling relationship. Anonymous target remotes have no ref
    /// to preserve in the projection; they represent commit-only target
    /// context such as `extra_target_commit_id` or a persisted workspace target
    /// commit.
    fn is_anonymous_integrated_target_context(&self) -> bool {
        matches!(self.role, SeedRole::TargetRemote) && self.ref_name.is_none()
    }

    /// Return whether this anonymous integrated target seed is auxiliary
    /// traversal context.
    ///
    /// Anonymous target remotes can be provided explicitly by callers and
    /// usually remain normal traversal seeds. The `auxiliary_integrated_seed_ids`
    /// set records the anonymous integrated targets that normalization derived
    /// from metadata or options such as `extra_target_commit_id`; those seeds act
    /// as mergeable limits/context and should be ordered or deduplicated as
    /// auxiliary work rather than as user-visible roots.
    ///
    /// If an anonymous target points to the same commit as a named target ref,
    /// normalization collapses it into the named seed.
    fn is_auxiliary_integrated_seed(
        &self,
        auxiliary_integrated_seed_ids: &BTreeSet<gix::ObjectId>,
    ) -> bool {
        self.is_anonymous_integrated_target_context()
            && auxiliary_integrated_seed_ids.contains(&self.id)
    }

    /// Return whether this anonymous integrated target should reuse the named
    /// target traversal seed for the same commit.
    ///
    /// The anonymous seed only contributes commit-level target context
    /// (seeds with [SeedRole::TargetRemote]). It does not need its own segment or
    /// queue item when a named target ref already points at that commit,
    /// and keeping both can make the anonymous seed own the commit while
    /// the named ref is left as a duplicate empty segment.
    fn collapses_into_named_integrated_target(
        &self,
        named_integrated_target_ids: &BTreeSet<gix::ObjectId>,
    ) -> bool {
        self.is_anonymous_integrated_target_context()
            && named_integrated_target_ids.contains(&self.id)
    }
}

/// The role a resolved traversal seed plays when constructing a graph.
///
/// Roles decide the initial [`CommitFlags`] and `Limit` goals used by the
/// walk. The explicit entrypoint is the shared goal: reachable and integrated
/// seeds seek connection to it by walking history until they encounter the entrypoint's
/// propagated goal flag.
///
/// Remote-tracking seeds are not modeled as explicit [`SeedRole`] values. They
/// are discovered during traversal from refs found at visited commits and their
/// configured or deduced remote-tracking branches. When such a remote seed is
/// queued, it receives an indirect goal for the local commit where it was
/// discovered, while that local side receives a goal for the remote seed. This
/// reciprocal goal setup lets remote and local tracking histories converge until
/// the graph can connect them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum SeedRole {
    /// A non-remote seed that should be traversed and related to the entrypoint.
    ///
    /// This seed marks all commits it traverses with [`CommitFlags::NotInRemote`].
    #[default]
    Reachable,
    /// The workspace ref itself, paired with workspace metadata on [`Seed`].
    ///
    /// This marks commits as in-workspace with [`CommitFlags::InWorkspace`].
    Workspace,
    /// A branch from a stack listed in workspace metadata.
    ///
    /// Its current ref tip should be traversed even if it is not reachable from
    /// the workspace commit.
    WorkspaceStackBranch {
        /// Ref name from workspace metadata to use for segment naming if the
        /// initial segment cannot infer an unambiguous ref from the seed commit.
        ///
        /// This is not [`Seed::ref_name`] because that field forces the initial
        /// segment to use the supplied name. Workspace stack branches should
        /// still allow normal ref discovery to pick an unambiguous local branch
        /// at the seed commit, or to leave the segment anonymous when local
        /// naming is ambiguous. The desired name is only a fallback for
        /// remote-only stack refs that cannot be discovered by local-branch
        /// disambiguation.
        ///
        /// Note that [Seed::id] is assumed to be the peeled commit that this
        /// ref points to.
        desired_ref_name: gix::refs::FullName,
    },
    /// A target/integration seed whose reachable history is considered integrated,
    /// and that reachable/unintegrated seeds want to connect with.
    ///
    /// This seed receives [`CommitFlags::Integrated`] and an indirect goal for
    /// the entrypoint commit with no extra allowance once that goal is found. It
    /// walks just far enough to connect target history to the entrypoint's
    /// ancestry.
    TargetRemote,
    /// The local branch that tracks an integrated target branch.
    ///
    /// It receives a goal for the target and later provides the segment id that
    /// lets the target segment point back to its local sibling.
    TargetLocal {
        /// The expected local tracking ref name used to verify whether the
        /// segment that normal ref discovery created is actually the local side
        /// of this target.
        ///
        /// This is not [`Seed::ref_name`] because that would force the segment
        /// to use this name and bypass ambiguity checks. If multiple local
        /// branches point to the same commit, or discovery chooses a different
        /// unambiguous name, the target should still get the local goal but not
        /// a direct sibling link.
        ///
        /// This matters when the target's local tracking branch shares its seed
        /// with another local branch, such as a workspace stack branch or a
        /// second branch with metadata. In that state, the segment may
        /// represent that other branch or stay anonymous; linking it as the
        /// target local side would make target ahead/behind and remote-reachability
        /// queries treat the wrong segment as the tracking branch.
        local_ref_name: gix::refs::FullName,
    },
}

/// Access
impl SeedRole {
    /// Whether this role represents integrated history.
    pub fn is_integrated(&self) -> bool {
        matches!(self, SeedRole::TargetRemote)
    }
}

/// A local branch ref and the commit it points to, when it tracks a workspace
/// target ref.
pub(crate) type LocalTrackingTip = (gix::refs::FullName, gix::ObjectId);

/// A workspace target ref, its commit, and optionally the local branch tracking it.
pub(crate) type WorkspaceTargetTip = (gix::refs::FullName, gix::ObjectId, Option<LocalTrackingTip>);

/// The complete pre-traversal plan derived from either explicit seeds or
/// workspace metadata.
///
/// [`queue_initial_seeds()`] consumes this value to create graph *segments*, seed
/// the traversal queue, and provide the auxiliary ref and remote information
/// needed by the traversal and the graph build.
///
/// This means that each of these seed *will get its own possibly empty* graph segment.
struct InitialSeeds {
    /// Ordered traversal roots to turn into segments and queue items.
    seeds: Vec<Seed>,
    /// Workspace commits used to ensure commits remain owned by the workspace
    /// roots that introduced them.
    workspace_seeds: Vec<gix::ObjectId>,
    /// Workspace ref names that should be included while collecting refs by
    /// prefix, even when they are not reachable from the entrypoint yet.
    workspace_ref_names: Vec<gix::refs::FullName>,
    /// Remote target refs that were already scheduled as initial integrated
    /// seeds.
    ///
    /// Workspace traversals seed this list from the project metadata target
    /// ref. Explicit traversal seeds the same list from integrated seed ref
    /// names. During traversal, `try_queue_remote_tracking_branches()` uses
    /// it to avoid queueing those target refs again when local branch refs
    /// point at them as upstreams.
    // TODO: could this be removed in favor os using `Graph::seeds`?
    target_refs: Vec<gix::refs::FullName>,
    /// Remote names to try when a local branch has no configured upstream.
    ///
    /// `lookup_remote_tracking_branch_or_deduce_it()` first asks Git for the
    /// branch's configured remote-tracking ref. If none exists, it tries each
    /// name here by constructing `refs/remotes/<remote>/<local-short-name>` and
    /// using it only if that ref exists and is not already configured for
    /// another branch.
    symbolic_remote_names: Vec<String>,
    /// Whether metadata-derived workspace/target seeds should be front-loaded
    /// into the traversal queue after their segments are created.
    frontload_workspace_related_seeds: bool,
    /// Target remote/local tracking relationships inferred from seed refs and
    /// repository branch configuration.
    ///
    /// These links are needed before traversal starts because target and local
    /// tracking seeds may point to the same commit, or may be reached in either
    /// order. Queueing uses this map to delay the target side until the local
    /// side has a segment and goal, then links both segments as siblings before
    /// their commits can be claimed by unrelated stack or reachable seeds. That
    /// keeps target ownership, ahead/behind, and remote-reachability queries
    /// anchored to the intended target/local pair.
    target_local_links: TargetLocalLinks,
    /// Anonymous target-remote seeds that are auxiliary traversal context rather
    /// than primary target refs.
    auxiliary_integrated_seed_ids: BTreeSet<gix::ObjectId>,
}

/// Bidirectional lookup between target remote refs and their local tracking refs.
#[derive(Default)]
struct TargetLocalLinks {
    /// Local tracking ref by target remote ref.
    local_by_target: BTreeMap<gix::refs::FullName, gix::refs::FullName>,
    /// Target remote ref by local tracking ref.
    target_by_local: BTreeMap<gix::refs::FullName, gix::refs::FullName>,
}

/// A way to define information to be served from memory, instead of from the underlying data source, when
/// [initializing](Graph::from_tip()) the graph.
#[derive(Debug, Default, Clone)]
pub struct Overlay {
    entrypoint: Entrypoint,
    nonoverriding_references: Vec<gix::refs::Reference>,
    overriding_references: Vec<gix::refs::Reference>,
    /// A list of references that should not be picked up anymore in the
    /// re-traversal.
    ///
    /// For example, if the `but_rebase::graph_rebase::Editor` converts a
    /// `Reference` step to a `None` step which is the equivalent of running
    /// `git update-ref -d`, it should no longer be part of the [`Graph`], so we
    /// would list the particular reference as a dropped reference.
    dropped_references: Vec<gix::refs::FullName>,
    meta_branches: Vec<(gix::refs::FullName, ref_metadata::Branch)>,
    branch_stack_orders: Vec<Vec<gix::refs::FullName>>,
    workspace: Option<(gix::refs::FullName, ref_metadata::Workspace)>,
}

/// Options for use in [`Graph::from_head()`] and [`Graph::from_tip()`].
#[derive(Default, Debug, Clone)]
pub struct Options {
    /// Associate tag references with commits.
    ///
    /// If `false`, tags are not collected.
    pub collect_tags: bool,
    /// The (soft) maximum number of commits we should traverse.
    /// Workspaces with a target branch automatically have unlimited traversals as they rely on the target
    /// branch to eventually stop the traversal.
    ///
    /// If `None`, there is no limit, which typically means that when lacking a workspace, the traversal
    /// will end only when no commit is left to traverse.
    /// `Some(0)` means nothing but the first commit is going to be returned, but it should be avoided.
    ///
    /// Note that this doesn't affect the traversal of integrated commits, which is always stopped once there
    /// is nothing interesting left to traverse.
    ///
    /// Also note: This is a hint and not an exact measure, and it's always possible to receive a more commits
    /// for various reasons, for instance the need to let remote branches find their local branch independently
    /// of the limit.
    pub commits_limit_hint: Option<usize>,
    /// A list of the last commits of partial segments previously returned that reset the amount of available
    /// commits to traverse back to `commit_limit_hint`.
    /// Imagine it like a gas station that can be chosen to direct where the commit-budge should be spent.
    pub commits_limit_recharge_location: Vec<gix::ObjectId>,
    /// As opposed to the limit-hint, if not `None` we will stop queuing new commits after pretty much this many
    /// commits have been seen.
    ///
    /// This is a last line of defense against runaway traversals and for now it's recommended to set it to a high
    /// but manageable value. Note that depending on the commit-graph, we may need more commits to find the local branch
    /// for a remote branch, leaving remote branches unconnected. Commits that are already queued are still processed so
    /// their existing graph connections can be completed.
    ///
    /// Due to multiple paths being taken, more commits may be queued (which is what's counted here) than actually
    /// end up in the graph, so usually one will see many less.
    pub hard_limit: Option<usize>,
    /// Provide the commit that should act like the tip of an additional target reference,
    /// just as if it was set by one of the workspaces.
    /// Everything it touches will be considered integrated, and it can be used
    /// to extend the border of the workspace. Typically, it's a past position
    /// of an existing target, or a target chosen by the user.
    pub extra_target_commit_id: Option<gix::ObjectId>,
}

/// Presets
impl Options {
    /// Return options that won't traverse the whole graph if there is no workspace, but will show
    /// more than enough commits by default.
    pub fn limited() -> Self {
        Options {
            collect_tags: false,
            commits_limit_hint: Some(300),
            ..Default::default()
        }
    }
}

/// Builder
impl Options {
    /// Set the maximum amount of commits that each seed's walk may traverse, but that's less important
    /// than building consistent, connected graphs.
    pub fn with_limit_hint(mut self, limit: usize) -> Self {
        self.commits_limit_hint = Some(limit);
        self
    }

    /// Set a hard limit for the amount of commits to traverse. Even though it may be off by a couple, it's not dependent
    /// on any additional logic.
    ///
    /// ### Warning
    ///
    /// This stops traversal early despite not having discovered all desired graph partitions, possibly leading to
    /// incorrect results. Ideally, this is not used.
    pub fn with_hard_limit(mut self, limit: usize) -> Self {
        self.hard_limit = Some(limit);
        self
    }

    /// Keep track of commits at which the traversal limit should be reset to the [`limit`](Self::with_limit_hint()).
    pub fn with_limit_extension_at(
        mut self,
        commits: impl IntoIterator<Item = gix::ObjectId>,
    ) -> Self {
        self.commits_limit_recharge_location.extend(commits);
        self
    }

    /// Set an additional integrated traversal seed.
    /// It's most useful for tests which want to affect the target of the workspace
    /// without the respective setup.
    /// Application code may use it to set global targets, to reduce the amount of
    /// commits in the workspace even if the entrypoint otherwise is the target branch.
    ///
    /// The commit is queued like an integrated target so traversal can connect
    /// the workspace to history that may otherwise be outside the ordinary
    /// target ref or workspace metadata. The seed is also kept as a seed of
    /// interest and re-resolved against the built graph so workspace projection
    /// can use it as a past target/base candidate.
    pub fn with_extra_target_commit_id(mut self, id: impl Into<gix::ObjectId>) -> Self {
        self.extra_target_commit_id = Some(id.into());
        self
    }
}

/// Lifecycle
impl Graph {
    /// Read the `HEAD` of `repo` and represent whatever is visible as a graph.
    ///
    /// See [`Self::from_tip()`] for details.
    pub fn from_head(
        repo: &gix::Repository,
        meta: &impl RefMetadata,
        project_meta: ProjectMeta,
        options: Options,
    ) -> anyhow::Result<Self> {
        let head = repo.head()?;
        // The dispatch lives in `from_tip` (which every case below delegates to):
        // a checkout inside a managed workspace — including HEAD on the workspace ref itself —
        // builds via the managed builder, everything else via the non-managed one.
        let mut is_detached = false;
        let (seed, maybe_name) = match head.kind {
            gix::head::Kind::Unborn(ref_name) => {
                let mut graph = Graph {
                    project_meta,
                    ..Default::default()
                };
                // It's OK to default-initialise this here as overlays are only used when redoing
                // the traversal.
                let (_repo, meta, _entrypoint) = Overlay::default().into_parts(repo, meta);
                let wt_by_branch = {
                    // Assume linked worktrees are never unborn!
                    let mut m = BTreeMap::new();
                    m.insert(
                        ref_name.clone(),
                        vec![crate::Worktree {
                            kind: crate::WorktreeKind::Main,
                            owned_by_repo: true,
                        }],
                    );
                    m
                };
                graph.insert_segment_set_entrypoint(branch_segment_from_name_and_meta(
                    Some((ref_name, None)),
                    &meta,
                    None,
                    &wt_by_branch,
                )?);
                return Ok(graph);
            }
            gix::head::Kind::Detached { target, peeled } => {
                is_detached = true;
                (peeled.unwrap_or(target).attach(repo), None)
            }
            gix::head::Kind::Symbolic(existing_reference) => {
                let mut existing_reference = existing_reference.attach(repo);
                let seed = existing_reference.peel_to_id()?;
                (seed, Some(existing_reference.inner.name))
            }
        };

        let mut graph = Self::from_tip(seed, maybe_name, meta, project_meta, options)?;
        if is_detached {
            graph.detach_entrypoint_segment()?;
        }
        Ok(graph)
    }
    /// Produce a minimal but usable representation of the commit-graph reachable from the commit at `seed` such the returned instance
    /// can represent everything that's observed, without losing information.
    /// `ref_name` is assumed to point to `seed` if given.
    ///
    /// `meta` is used to learn more about the encountered references, and `options` is used for additional configuration.
    ///
    /// ### Features
    ///
    /// * discover a Workspace on the fly based on `meta`-data.
    /// * support the notion of a branch to integrate with, the *target*
    ///     - *target* branches consist of a local and remote tracking branch, and one can be ahead of the other.
    ///     - workspaces are relative to the local tracking branch of the target.
    ///     - options contain an [`extra_target_commit_id`](Options::extra_target_commit_id) for an additional target location.
    /// * remote tracking branches are seen in relation to their branches.
    /// * the graph of segments assigns each reachable commit to exactly one segment
    /// * the segments form a small owned graph tailored to this crate, not a third-party library
    ///     - It maintains information about the intended connections, so modifications afterward will show
    ///       in debugging output if edges are now in violation of this constraint.
    ///
    /// ### Rules/Invariants
    ///
    /// These rules should help to create graphs and segmentations that feel natural and are desirable to the user,
    /// while avoiding traversing the entire commit-graph all the time.
    /// Change the rules as you see fit to accomplish this.
    ///
    /// * Traversal is seeded from [`Seed`]s. Workspace metadata traversal first
    ///   resolves metadata into seeds, then follows the same path as callers
    ///   passing explicit seeds.
    /// * Explicit seeds must contain exactly one entrypoint, must not contain
    ///   duplicate traversal seeds, and any named seed must have a ref that
    ///   resolves to its commit id. A traversal seed is the commit id, the
    ///   traversal role, and whether that seed is the entrypoint; naming,
    ///   metadata, detached presentation, and queue position do not make it
    ///   useful to enqueue the same seed twice.
    /// * Multiple seeds with different [roles](SeedRole) may point to the same commit id,
    ///   as multiple refs can name the same commit.
    /// * A detached seed must be the entrypoint and cannot carry a ref name.
    /// * The entrypoint always causes the start of a [`crate::Segment`].
    /// * Tips discovered from workspace metadata preserve their queue order.
    ///   Explicit seeds without a custom queue position are normalized into
    ///   deterministic traversal order: integrated and target seeds first,
    ///   reachable/workspace seeds next, and the entrypoint last.
    /// * A commit can be governed by multiple workspaces.
    /// * As workspaces and entrypoints "grow" together, we don't know anything
    ///   about workspaces until the very end, or when two partitions of commits
    ///   touch. This means we can't make decisions based on
    ///   [flags](CommitFlags) until the traversal is finished.
    /// * Segments are named if their first commit has a single local branch
    ///   pointing to it, or a branch that otherwise can be disambiguated.
    /// * Anonymous segments are created if their name is ambiguous.
    /// * Anonymous segments are created if another segment connects to a commit
    ///   that it contains that is not the first one.
    ///    - This means, all connections go *from the last commit in a segment to the first commit in another segment*.
    /// * Stacks and branches stored in the *workspace metadata* are relevant only if they
    ///   become seeds backed by an existing branch.
    /// * Remote tracking branches are picked up during traversal for any ref
    ///   that we reached through traversal.
    ///     - Remote tracking branches are discovered only for refs encountered
    ///       during traversal. Segments minted later during the graph build,
    ///       especially virtual or empty segments, do not cause additional remote
    ///       traversal.
    ///     - Remote tracking branches never take commits that are already owned.
    /// * The traversal is cut short when only integrated seeds remain.
    /// * The traversal is always as long as it needs to be to fully reconcile
    ///   possibly disjoint branches, despite this sometimes costing some time
    ///   when the remote is far ahead in a huge repository.
    #[instrument(name = "Graph::from_tip", level = "trace", skip_all, fields(seed = ?seed, ref_name), err(Debug))]
    pub fn from_tip(
        seed: gix::Id<'_>,
        ref_name: impl Into<Option<gix::refs::FullName>>,
        meta: &impl RefMetadata,
        project_meta: ProjectMeta,
        options: Options,
    ) -> anyhow::Result<Self> {
        let repo = seed.repo;
        let seed = seed.detach();
        let ref_name = ref_name.into();
        // Build from a CommitGraph: inside a managed workspace with an entrypoint split, else via
        // the non-managed builder. A workspace-ref seed is the plain from_head case (no explicit
        // entrypoint).
        let is_ws_tip = ref_name
            .as_ref()
            .is_some_and(|r| but_core::is_workspace_ref_name(r.as_ref()));
        let (entrypoint, entrypoint_ref) = if is_ws_tip {
            (None, None)
        } else {
            (Some(seed), ref_name.clone())
        };
        if let Some(graph) = crate::graph_from_repository(
            repo,
            meta,
            entrypoint,
            entrypoint_ref,
            project_meta.clone(),
            options.clone(),
            Overlay::default(),
        )? {
            return Ok(graph);
        }
        // No managed workspace, or the entrypoint is outside it: the non-managed builder.
        crate::graph_from_repository_unmanaged(
            repo,
            meta,
            seed,
            ref_name,
            project_meta,
            options,
            Overlay::default(),
        )
    }

    /// Produce a graph from already resolved seeds and their traversal roles.
    ///
    /// This is useful for callers that already know the commits they want to
    /// relate, or whose seeds are not represented by durable repository refs or
    /// workspace metadata.
    ///
    /// `repo` provides commit objects, refs, remotes, worktrees, and optional
    /// commit-graph acceleration for traversal.
    /// `seeds` provides the resolved commits and their traversal roles. It must
    /// contain exactly one seed whose [`Seed::is_entrypoint`] flag is set.
    /// `meta` provides branch metadata for any refs encountered while walking.
    /// `options` controls tag collection, traversal limits, additional
    /// integrated seeds, and graph-build behavior.
    pub fn from_seeds(
        repo: &gix::Repository,
        seeds: impl IntoIterator<Item = Seed>,
        meta: &impl RefMetadata,
        project_meta: ProjectMeta,
        options: Options,
    ) -> anyhow::Result<Self> {
        let seeds: Vec<_> = seeds.into_iter().collect();
        // Build from a CommitGraph derived from the same seeds traversal.
        crate::graph_from_repository_seeds(repo, meta, seeds, project_meta, options)
    }
    /// Walk from a checked-out tip, seeded like [`Self::from_tip`],
    /// accumulating commits directly (see `walker`).
    pub(crate) fn walk_from_tip(
        repo: &gix::Repository,
        seed: gix::ObjectId,
        ref_name: Option<gix::refs::FullName>,
        meta: &impl RefMetadata,
        project_meta: ProjectMeta,
        options: Options,
        overlay: Overlay,
    ) -> anyhow::Result<walker::WalkOutcome> {
        let (overlay_repo, overlay_meta, _entrypoint) = overlay.into_parts(repo, meta);
        let seeds = initial_seeds_from_workspace_metadata(
            &overlay_repo,
            &overlay_meta,
            seed,
            ref_name.as_ref(),
            &project_meta,
            options.extra_target_commit_id,
        )?;
        walker::traverse(
            &overlay_repo,
            seeds,
            &overlay_meta,
            project_meta,
            options,
            ref_name,
        )
    }

    /// Walk from explicit seeds, like [`Self::from_seeds`], accumulating
    /// commits directly (see `walker`).
    pub(crate) fn walk_from_seeds(
        repo: &gix::Repository,
        seeds: Vec<Seed>,
        meta: &impl RefMetadata,
        project_meta: ProjectMeta,
        options: Options,
        overlay: Overlay,
    ) -> anyhow::Result<walker::WalkOutcome> {
        let (overlay_repo, overlay_meta, _entrypoint) = overlay.into_parts(repo, meta);
        walker::traverse(
            &overlay_repo,
            seeds,
            &overlay_meta,
            project_meta,
            options,
            None,
        )
    }

    /// Take the ref-info from a named segment and put it back onto the first commit
    /// where it pointed to before it was lifted up.
    ///
    /// Graph traversal eagerly names segments from refs pointing at their
    /// first commit. Detached entrypoints keep those refs on the commit, but
    /// the entrypoint segment itself must stay anonymous.
    pub(crate) fn detach_entrypoint_segment(&mut self) -> anyhow::Result<()> {
        let sidx = self
            .entrypoint
            .context("BUG: entrypoint is set after first traversal")?
            .0;
        let s = &mut self[sidx];
        if let Some((rn, first_commit)) = s
            .commits
            .first_mut()
            .and_then(|first_commit| s.ref_info.take().map(|rn| (rn, first_commit)))
        {
            first_commit.refs.push(rn);
        }
        Ok(())
    }

    /// Repeat the traversal that generated this graph using `repo` and `meta`, but allow to set an in-memory
    /// `overlay` to amend the data available from `repo` and `meta`.
    /// This way, one can see this graph as it will be in the future once the changes to `repo` and `meta` are actually made.
    pub fn redo_traversal(
        &self,
        repo: &gix::Repository,
        meta: &impl RefMetadata,
        overlay: Overlay,
    ) -> anyhow::Result<Self> {
        let overlay_for_flip = overlay.clone();
        let (repo, meta, entrypoint) = overlay.into_parts(repo, meta);
        let (seed, ref_name) = match entrypoint {
            Some(t) => t,
            None => {
                let (entrypoint_sidx, commit) = self
                    .entrypoint
                    .context("BUG: entrypoint must always be set")?;
                let entrypoint_segment = self
                    .segment(entrypoint_sidx)
                    .context("BUG: entrypoint segment must be present")?;
                let mut ref_name = entrypoint_segment.ref_info.clone().map(|ri| ri.ref_name);
                let seed = if let Some(name) = ref_name.as_ref() {
                    match repo.try_find_reference(name.as_ref())? {
                        Some(mut reference) => Some(reference.peel_to_id()?.detach()),
                        None => {
                            // The previous traversal may have had a named entrypoint, but
                            // this overlay can drop that ref. If so, don't carry a stale
                            // entrypoint_ref override into the new traversal; it would fail
                            // validation instead of re-traversing from the remembered commit.
                            ref_name = None;
                            None
                        }
                    }
                } else {
                    None
                };
                let seed = seed
                    .or_else(|| commit.object_id())
                    .context(
                        "BUG: entrypoint must either remember the original commit id or have a resolvable ref",
                    )?;
                (seed, ref_name)
            }
        };
        // The same dispatch as `from_tip`, with the overlay served from memory by
        // the builders.
        let is_ws_tip = ref_name
            .as_ref()
            .is_some_and(|r| but_core::is_workspace_ref_name(r.as_ref()));
        let (flip_ep, flip_ep_ref) = if is_ws_tip {
            (None, None)
        } else {
            (Some(seed), ref_name.clone())
        };
        if let Some(graph) = crate::graph_from_repository(
            repo.for_attach_only(),
            meta.for_inner_only(),
            flip_ep,
            flip_ep_ref,
            self.project_meta.clone(),
            self.options.clone(),
            overlay_for_flip.clone(),
        )? {
            return Ok(graph);
        }
        // No managed workspace, or the entrypoint is outside it: the non-managed builder.
        crate::graph_from_repository_unmanaged(
            repo.for_attach_only(),
            meta.for_inner_only(),
            seed,
            ref_name,
            self.project_meta.clone(),
            self.options.clone(),
            overlay_for_flip,
        )
    }
}

/// Validate caller-provided traversal seeds before they seed graph traversal.
///
/// Explicit seeds must name exactly one entrypoint, must not contain duplicate
/// traversal seeds or repeated ref names, must keep detached entrypoints
/// unnamed, and any supplied ref name must resolve to the same commit id as its
/// seed.
fn validate_explicit_seeds<'a>(
    repo: &OverlayRepo<'_>,
    seeds: &'a [Seed],
    entrypoint_ref_override: Option<&gix::refs::FullName>,
) -> anyhow::Result<&'a Seed> {
    let mut entrypoints = seeds.iter().filter(|seed| seed.is_entrypoint);
    let entrypoint = entrypoints
        .next()
        .context("explicit traversal seeds require exactly one entrypoint")?;
    ensure!(
        entrypoints.next().is_none(),
        "explicit traversal seeds require exactly one entrypoint"
    );

    for (idx, seed) in seeds.iter().enumerate() {
        ensure!(
            !seed.is_detached || seed.is_entrypoint,
            "explicit detached seed must also be the entrypoint"
        );
        ensure!(
            !seed.is_detached || seed.ref_name.is_none(),
            "explicit detached entrypoint seed cannot have a ref name"
        );
        ensure!(
            !seed.is_entrypoint || matches!(seed.role, SeedRole::Reachable | SeedRole::Workspace),
            "explicit entrypoint seed must be reachable or workspace"
        );

        for previous in &seeds[..idx] {
            ensure!(
                !seeds_have_same_traversal(previous, seed),
                "explicit traversal seeds contain duplicate traversal seed {seed:?}"
            );
            if let Some(ref_name) = seed
                .ref_name
                .as_ref()
                .filter(|ref_name| previous.ref_name.as_ref() == Some(*ref_name))
            {
                bail!("explicit traversal seeds contain duplicate ref name {ref_name}");
            }
        }

        if let Some(ref_name) = seed.ref_name.as_ref() {
            validate_seed_ref(repo, ref_name, seed.id, "explicit traversal seed ref")?;
        }
    }

    if !entrypoint.is_detached
        && let Some(ref_name) = entrypoint_ref_override
    {
        validate_seed_ref(
            repo,
            ref_name,
            entrypoint.id,
            "explicit traversal entrypoint ref",
        )?;
    }

    Ok(entrypoint)
}

fn validate_seed_ref(
    repo: &OverlayRepo<'_>,
    ref_name: &gix::refs::FullName,
    tip_id: gix::ObjectId,
    context: &str,
) -> anyhow::Result<()> {
    let resolved_id = repo
        .try_find_reference(ref_name.as_ref())?
        .with_context(|| format!("{context} {ref_name} does not exist"))?
        .peel_to_id()?
        .detach();
    ensure!(
        resolved_id == tip_id,
        "{context} {ref_name} points to {resolved_id}, not {tip_id}"
    );
    Ok(())
}

/// Return whether two seeds would seed the same traversal work.
///
/// The traversal seed is the commit id, the traversal role, and whether the seed
/// is the entrypoint. Labels and presentation data like `ref_name`, metadata,
/// detached entrypoint mode, and caller order are intentionally ignored here:
/// they can affect naming, the graph build, or stable tie-breaking, but they
/// don't make it useful to enqueue the same commit with the same traversal
/// semantics twice.
fn seeds_have_same_traversal(previous: &Seed, seed: &Seed) -> bool {
    previous.id == seed.id
        && seeds_have_same_role(previous, seed)
        && previous.is_entrypoint == seed.is_entrypoint
}

/// Return whether two seeds have the same traversal role for deduplication.
///
/// [`SeedRole::TargetRemote`] is special because named and anonymous target
/// remotes with the same commit can have different responsibilities. A named
/// target remote represents a ref that may need its own segment,
/// metadata-derived target identity, and target/local sibling link. An
/// anonymous target remote represents commit-only target context, such as
/// `extra_target_commit_id` or a persisted target commit. Validation accepts
/// those two forms so callers can pass metadata-equivalent seeds directly;
/// normalization later collapses the anonymous form into the named seed if they
/// point to the same commit.
fn seeds_have_same_role(previous: &Seed, seed: &Seed) -> bool {
    match (&previous.role, &seed.role) {
        (SeedRole::TargetRemote, SeedRole::TargetRemote) => {
            previous.ref_name.is_some() == seed.ref_name.is_some()
        }
        _ => previous.role == seed.role,
    }
}

/// Build auxiliary traversal inputs from normalized seeds.
fn assemble_initial_seeds(
    repo: &OverlayRepo<'_>,
    mut seeds: Vec<Seed>,
    project_meta: &ProjectMeta,
    extra_target_commit_id: Option<gix::ObjectId>,
) -> InitialSeeds {
    let mut auxiliary_integrated_seed_ids = BTreeSet::new();
    if let Some(extra_target) = extra_target_commit_id {
        auxiliary_integrated_seed_ids.insert(extra_target);
        push_integrated_seed_once(&mut seeds, extra_target);
    }
    let frontload_workspace_related_seeds = has_workspace_related_seeds(&seeds);
    if frontload_workspace_related_seeds {
        auxiliary_integrated_seed_ids.extend(seeds.iter().filter_map(|seed| {
            seed.is_anonymous_integrated_target_context()
                .then_some(seed.id)
        }));
    }
    collapse_anonymous_integrated_seeds_into_named_targets(&mut seeds);
    let seeds = seeds_in_queue_order(seeds, &auxiliary_integrated_seed_ids);
    let workspace_seeds = seeds
        .iter()
        .filter(|seed| matches!(seed.role, SeedRole::Workspace))
        .map(|seed| seed.id)
        .collect();
    let workspace_ref_names = seeds
        .iter()
        .filter(|seed| matches!(seed.role, SeedRole::Workspace))
        .filter_map(|seed| seed.ref_name.clone())
        .collect();
    let include_seed_refs = !seeds
        .iter()
        .any(|seed| matches!(seed.metadata, Some(SegmentMetadata::Workspace(_))));
    let target_refs = target_refs_from_seeds(&seeds, project_meta, include_seed_refs);
    let symbolic_remote_names =
        symbolic_remote_names_from_seeds(repo, &seeds, project_meta, include_seed_refs);
    let target_local_links = target_local_links_from_seeds(repo, &seeds);

    InitialSeeds {
        seeds,
        workspace_seeds,
        workspace_ref_names,
        target_refs,
        symbolic_remote_names,
        frontload_workspace_related_seeds,
        target_local_links,
        auxiliary_integrated_seed_ids,
    }
}

/// Remove anonymous integrated target seeds that point to the same commit as a
/// named integrated target.
///
/// Workspace projection derives target context from target-remote seeds by graph
/// position, so a same-commit anonymous target does not contribute anything
/// once a named target ref covers that commit. Collapsing here keeps one
/// effective traversal seed and lets the named target segment own the commit.
fn collapse_anonymous_integrated_seeds_into_named_targets(seeds: &mut Vec<Seed>) {
    let named_integrated_target_ids = seeds
        .iter()
        .filter_map(|seed| {
            (matches!(seed.role, SeedRole::TargetRemote) && seed.ref_name.is_some())
                .then_some(seed.id)
        })
        .collect::<BTreeSet<_>>();
    seeds.retain(|seed| !seed.collapses_into_named_integrated_target(&named_integrated_target_ids));
}

/// Convert validated seeds into deterministic initial traversal roots.
///
/// The caller can provide explicit seeds in any order, but queue order still
/// matters because the first item that reaches a commit owns the segment for
/// that commit. This function recreates the ordering that metadata-derived
/// traversal would have produced for workspace seeds, while keeping the simpler
/// historical ordering for plain commit traversal.
///
/// The sort is intentionally heuristic: role priority establishes the broad
/// traversal shape, workspace metadata restores stack/branch order when it is
/// available, and stable tie-breakers make equivalent inputs independent of
/// caller order. For non-workspace traversals, equal-priority seeds keep caller
/// order so existing explicit traversal behavior stays predictable.
fn seeds_in_queue_order(
    seeds: Vec<Seed>,
    auxiliary_integrated_seed_ids: &BTreeSet<gix::ObjectId>,
) -> Vec<Seed> {
    let has_workspace_related_seeds = has_workspace_related_seeds(&seeds);
    let workspace_branch_order = workspace_branch_order_from_seeds(&seeds);
    let mut seeds: Vec<_> = seeds.into_iter().enumerate().collect();
    seeds.sort_by(|(a_idx, a), (b_idx, b)| {
        seed_queue_priority(
            a,
            has_workspace_related_seeds,
            auxiliary_integrated_seed_ids,
        )
        .cmp(&seed_queue_priority(
            b,
            has_workspace_related_seeds,
            auxiliary_integrated_seed_ids,
        ))
        .then_with(|| {
            seed_workspace_branch_order(a, &workspace_branch_order)
                .cmp(&seed_workspace_branch_order(b, &workspace_branch_order))
        })
        .then_with(|| {
            if has_workspace_related_seeds {
                seed_sort_name(a).cmp(&seed_sort_name(b))
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .then_with(|| {
            if has_workspace_related_seeds {
                a.id.cmp(&b.id)
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .then_with(|| a_idx.cmp(b_idx))
    });
    seeds.into_iter().map(|(_, seed)| seed).collect()
}

/// Return whether seed ordering has to emulate workspace metadata traversal.
///
/// Workspace, workspace-stack, and target-local seeds are not just additional
/// roots. Their relative order influences which segment owns a shared commit
/// and how the graph build mints virtual workspace and stack segments.
/// Detecting such seeds switches sorting from "mostly preserve caller order" to
/// "rebuild the metadata order deterministically".
fn has_workspace_related_seeds(seeds: &[Seed]) -> bool {
    seeds.iter().any(|seed| {
        matches!(
            seed.role,
            SeedRole::Workspace
                | SeedRole::TargetLocal { .. }
                | SeedRole::WorkspaceStackBranch { .. }
        ) || matches!(seed.metadata, Some(SegmentMetadata::Workspace(_)))
    })
}

/// Primary sort key for initial seeds.
///
/// This is the main heuristic. For workspace-related traversals we recreate
/// the metadata-derived segment creation order:
///
/// 1. A non-workspace reachable entrypoint first, if there is one.
/// 2. The workspace ref so it can become the traversal anchor.
/// 3. The integrated target ref, then its local tracking branch, so they can
///    be linked as siblings and agree on target ownership.
/// 4. Synthetic integrated targets, like extra target commits.
/// 5. Workspace stack branches, whose order is refined later from workspace
///    metadata.
/// 6. Other reachable roots.
///
/// For non-workspace traversals there is no metadata order to recover, so
/// integrated context still comes first, non-entry reachable roots follow, and
/// the entrypoint anchors the graph last. Synthetic integrated seeds remain
/// last because they are auxiliary limits, not primary user roots.
fn seed_queue_priority(
    seed: &Seed,
    has_workspace_related_seeds: bool,
    auxiliary_integrated_seed_ids: &BTreeSet<gix::ObjectId>,
) -> usize {
    if has_workspace_related_seeds {
        match &seed.role {
            SeedRole::Reachable if seed.is_entrypoint => 0,
            SeedRole::Workspace => 1,
            SeedRole::TargetRemote if seed.ref_name.is_some() => 2,
            SeedRole::TargetLocal { .. } => 3,
            SeedRole::TargetRemote
                if seed.is_auxiliary_integrated_seed(auxiliary_integrated_seed_ids) =>
            {
                4
            }
            SeedRole::TargetRemote => 2,
            SeedRole::WorkspaceStackBranch { .. } => 5,
            SeedRole::Reachable => 6,
        }
    } else {
        match &seed.role {
            SeedRole::TargetRemote
                if seed.is_auxiliary_integrated_seed(auxiliary_integrated_seed_ids) =>
            {
                3
            }
            SeedRole::TargetRemote => 0,
            SeedRole::TargetLocal { .. } => 0,
            SeedRole::Reachable | SeedRole::Workspace | SeedRole::WorkspaceStackBranch { .. } => {
                if seed.is_entrypoint {
                    2
                } else {
                    1
                }
            }
        }
    }
}

/// Recover stack-branch order from workspace metadata.
///
/// Workspace metadata stores the user-visible ordering of workspaces, stacks,
/// and branches. When explicit seeds are equivalent to metadata-derived seeds,
/// this order is the only reliable way to make scrambled input produce the same
/// graph and workspace projection as `from_tip()`.
///
/// The return value maps a branch ref name to the position where that branch
/// appears in workspace metadata. The value tuple is
/// `(workspace_order, stack_order, branch_order)`:
///
/// - `workspace_order` is the index of the workspace metadata seed after all
///   workspace metadata seeds have been sorted by their optional ref name. This
///   makes multi-workspace input deterministic even when the caller provided
///   seeds in a different order.
/// - `stack_order` is the zero-based index among stacks that are currently in
///   the workspace. Archived or otherwise inactive stacks are ignored and don't
///   consume an order slot.
/// - `branch_order` is the zero-based index of the branch within that stack's
///   branch list.
///
/// Branch refs not found in this map have no metadata-derived order and fall
/// back to later tie-breakers. If the same branch ref appears more than once,
/// the first metadata occurrence wins, matching the "first configured stack
/// owns the branch" behavior expected by workspace projection.
fn workspace_branch_order_from_seeds(
    seeds: &[Seed],
) -> BTreeMap<gix::refs::FullName, (usize, usize, usize)> {
    let mut workspaces: Vec<_> = seeds
        .iter()
        .filter_map(|seed| match seed.metadata.as_ref() {
            Some(SegmentMetadata::Workspace(data)) => Some((seed.ref_name.as_ref(), data)),
            Some(SegmentMetadata::Branch(_)) | None => None,
        })
        .collect();
    workspaces.sort_by_key(|(ref_name, _)| *ref_name);

    let mut out = BTreeMap::new();
    for (workspace_order, (_ref_name, data)) in workspaces.into_iter().enumerate() {
        for (stack_order, stack) in data
            .stacks
            .iter()
            .filter(|stack| stack.is_in_workspace())
            .enumerate()
        {
            for (branch_order, branch) in stack.branches.iter().enumerate() {
                out.entry(branch.ref_name.clone()).or_insert((
                    workspace_order,
                    stack_order,
                    branch_order,
                ));
            }
        }
    }
    out
}

/// Return the metadata order for a workspace stack branch seed.
///
/// Only `WorkspaceStackBranch` seeds participate in this secondary ordering.
/// Other roles intentionally return `None` so their relative order is governed
/// by the primary role priority and later tie-breakers.
fn seed_workspace_branch_order(
    seed: &Seed,
    workspace_branch_order: &BTreeMap<gix::refs::FullName, (usize, usize, usize)>,
) -> Option<(usize, usize, usize)> {
    match &seed.role {
        SeedRole::WorkspaceStackBranch { desired_ref_name } => {
            workspace_branch_order.get(desired_ref_name).copied()
        }
        SeedRole::Reachable
        | SeedRole::Workspace
        | SeedRole::TargetRemote
        | SeedRole::TargetLocal { .. } => None,
    }
}

/// Stable name tie-breaker used only in workspace-related sorting.
///
/// After role priority and metadata branch order, seeds may still be equivalent
/// from the traversal's point of view. Sorting by the ref that will name or
/// identify the segment keeps explicit workspace-seed input order irrelevant.
/// For non-workspace traversals this helper is deliberately ignored so equal
/// priorities preserve the caller's order instead.
fn seed_sort_name(seed: &Seed) -> Option<String> {
    match &seed.role {
        SeedRole::WorkspaceStackBranch { desired_ref_name } => {
            Some(desired_ref_name.as_bstr().to_string())
        }
        SeedRole::TargetLocal { local_ref_name } => Some(local_ref_name.as_bstr().to_string()),
        SeedRole::Reachable | SeedRole::Workspace | SeedRole::TargetRemote => {
            seed.ref_name.as_ref().map(|ref_name| ref_name.to_string())
        }
    }
}

/// Discover workspaces, targets, local tracking branches, and workspace stack
/// branch refs and turn them into initial traversal seeds.
fn initial_seeds_from_workspace_metadata<T: RefMetadata>(
    repo: &OverlayRepo<'_>,
    meta: &OverlayMetadata<'_, T>,
    entrypoint: gix::ObjectId,
    entrypoint_ref: Option<&gix::refs::FullName>,
    project_meta: &ProjectMeta,
    extra_target_commit_id: Option<gix::ObjectId>,
) -> anyhow::Result<Vec<Seed>> {
    let workspaces = obtain_workspace_infos(repo, entrypoint_ref.map(|rn| rn.as_ref()), meta)?;
    let tip_ref_matches_ws_ref = workspaces
        .iter()
        .find_map(|(ws_tip, ws_rn, _)| (Some(ws_rn) == entrypoint_ref).then_some(ws_tip));

    let mut seeds = Vec::new();
    let mut workspace_metas = Vec::new();
    let mut additional_target_commits = Vec::new();
    let mut queued_ids = Vec::new();

    match tip_ref_matches_ws_ref {
        None => {
            // We don't name the seed of the entrypoint as we want the segment
            // naming to be handled by seeds created from metadata.
            seeds.push(Seed::entrypoint(entrypoint, None));
            queued_ids.push(entrypoint);
        }
        Some(ws_tip) => {
            ensure!(
                *ws_tip == entrypoint,
                format!(
                    "BUG:: {entrypoint_ref:?} points to {ws_tip}, but the caller claimed it points to {entrypoint}"
                )
            );
        }
    }

    for (ws_tip, ws_ref, ws_meta) in workspaces {
        workspace_metas.push(ws_meta.clone());
        additional_target_commits.extend(project_meta.target_commit_id);
        seeds.push(
            Seed::new(ws_tip)
                .with_ref_name(Some(ws_ref.clone()))
                .with_role(SeedRole::Workspace)
                .with_metadata(SegmentMetadata::Workspace(ws_meta.clone()))
                .with_is_entrypoint(Some(&ws_ref) == entrypoint_ref),
        );

        let target = if let Some((target_ref, target_ref_id, local_info)) =
            workspace_target_tip(repo, project_meta.target_ref.as_ref())?
        {
            let local_info =
                local_info.filter(|(_local_ref_name, local_tip)| !queued_ids.contains(local_tip));
            seeds.push(
                Seed::new(target_ref_id)
                    .with_ref_name(Some(target_ref))
                    .with_role(SeedRole::TargetRemote),
            );
            if let Some((local_ref_name, local_tip)) = local_info.clone() {
                seeds
                    .push(Seed::new(local_tip).with_role(SeedRole::TargetLocal { local_ref_name }));
            }
            Some((
                target_ref_id,
                local_info.map(|(_local_ref_name, local_tip)| local_tip),
            ))
        } else {
            None
        };
        queued_ids.push(ws_tip);
        if let Some((target_ref_id, local_tip)) = target {
            queued_ids.push(target_ref_id);
            if let Some(local_tip) = local_tip {
                queued_ids.push(local_tip);
            }
        }
    }

    if let Some(extra_target) = extra_target_commit_id {
        push_integrated_seed_once(&mut seeds, extra_target);
    }

    for target_commit_id in additional_target_commits {
        // These are possibly from metadata, and thus might not exist (anymore). Ignore if that's the case.
        if let Err(err) = repo.find_commit(target_commit_id) {
            tracing::warn!(
                ?target_commit_id,
                ?err,
                "Ignoring stale target commit id as it didn't exist"
            );
            continue;
        }
        // We don't really have a place to store the segment index of the segment owning the target commit
        // so we will re-acquire it later when building the workspace projection.
        push_integrated_seed_once(&mut seeds, target_commit_id);
    }

    // Queue workspace stack branch refs that may have advanced since the
    // workspace commit was written, and thus would not be reached from that
    // commit alone.
    for ws_metadata in workspace_metas {
        for segment in ws_metadata
            .stacks
            .into_iter()
            .filter(|s| s.is_in_workspace())
            .flat_map(|s| s.branches.into_iter())
        {
            let Some(segment_tip) = repo
                .try_find_reference(segment.ref_name.as_ref())?
                .map(|mut r| r.peel_to_id())
                .transpose()?
            else {
                continue;
            };
            push_seed_once(
                &mut seeds,
                Seed::new(segment_tip.detach()).with_role(SeedRole::WorkspaceStackBranch {
                    desired_ref_name: segment.ref_name,
                }),
            );
        }
    }

    Ok(seeds)
}

fn push_integrated_seed_once(seeds: &mut Vec<Seed>, id: gix::ObjectId) {
    let seed = Seed::new(id).with_role(SeedRole::TargetRemote);
    push_seed_once(seeds, seed);
}

fn push_seed_once(seeds: &mut Vec<Seed>, seed: Seed) {
    if !seeds
        .iter()
        .any(|existing| seeds_have_same_traversal(existing, &seed))
    {
        seeds.push(seed);
    }
}

/// Resolve a workspace target ref and, when possible, its local tracking branch
/// tip.
pub(crate) fn workspace_target_tip(
    repo: &OverlayRepo<'_>,
    target_ref: Option<&gix::refs::FullName>,
) -> anyhow::Result<Option<WorkspaceTargetTip>> {
    let Some(target_ref) = target_ref else {
        return Ok(None);
    };
    let target_ref_id = match try_refname_to_id(repo, target_ref.as_ref()).map_err(|err| {
        tracing::warn!("Ignoring non-existing target branch {target_ref}: {err}");
        err
    }) {
        Ok(Some(target_ref_id)) => target_ref_id,
        Ok(None) | Err(_) => return Ok(None),
    };
    let local_info = repo
        .upstream_branch_and_remote_for_tracking_branch(target_ref.as_ref())
        .ok()
        .flatten()
        .and_then(|(local_tracking_name, _remote_name)| {
            let target_local_tip = try_refname_to_id(repo, local_tracking_name.as_ref()).ok()??;
            Some((local_tracking_name, target_local_tip))
        });
    Ok(Some((target_ref.clone(), target_ref_id, local_info)))
}

/// Return remote target refs that are already represented by initial seeds.
///
/// The result is passed to remote-tracking discovery so it does not queue a
/// target ref a second time when walking a local branch that tracks it.
/// Workspace traversals get this from the project metadata target ref, which
/// is where their target lives now. Explicit traversals have no workspace
/// discovery source, so named integrated seeds may also act as target refs
/// when `include_integrated_seed_refs` is set.
fn target_refs_from_seeds(
    seeds: &[Seed],
    project_meta: &ProjectMeta,
    include_integrated_seed_refs: bool,
) -> Vec<gix::refs::FullName> {
    let has_workspace_metadata_seed = seeds
        .iter()
        .any(|seed| matches!(seed.metadata, Some(SegmentMetadata::Workspace(_))));
    let mut target_refs: Vec<_> = seeds
        .iter()
        .filter(|seed| include_integrated_seed_refs && seed.role.is_integrated())
        .filter_map(|seed| seed.ref_name.clone())
        .chain(
            has_workspace_metadata_seed
                .then(|| project_meta.target_ref.clone())
                .flatten(),
        )
        .collect();
    target_refs.sort();
    target_refs.dedup();
    target_refs
}

/// Infer target remote/local tracking links without exposing correlation ids on
/// public seeds.
///
/// The target side is represented by a named [`SeedRole::TargetRemote`] seed. The
/// local side is represented by a [`SeedRole::TargetLocal`] seed whose
/// `local_ref_name` matches the local branch configured to track that remote
/// target ref. If either side is absent, the seeds still participate in
/// traversal but no sibling link is prepared up front.
fn target_local_links_from_seeds(repo: &OverlayRepo<'_>, seeds: &[Seed]) -> TargetLocalLinks {
    let remote_target_refs: Vec<_> = seeds
        .iter()
        .filter(|seed| matches!(seed.role, SeedRole::TargetRemote))
        .filter_map(|seed| seed.ref_name.clone())
        .collect();
    let local_refs: BTreeSet<_> = seeds
        .iter()
        .filter_map(|seed| match &seed.role {
            SeedRole::TargetLocal { local_ref_name } => Some(local_ref_name.clone()),
            SeedRole::Reachable
            | SeedRole::Workspace
            | SeedRole::WorkspaceStackBranch { .. }
            | SeedRole::TargetRemote => None,
        })
        .collect();

    let mut links = TargetLocalLinks::default();
    for target_ref in remote_target_refs {
        let Some((local_ref, _remote_name)) = repo
            .upstream_branch_and_remote_for_tracking_branch(target_ref.as_ref())
            .ok()
            .flatten()
        else {
            continue;
        };
        if !local_refs.contains(&local_ref) {
            continue;
        }
        links
            .local_by_target
            .insert(target_ref.clone(), local_ref.clone());
        links.target_by_local.insert(local_ref, target_ref);
    }
    links
}

/// Collect symbolic remote names implied by seed refs, workspace target refs,
/// workspace `push_remote` settings, and stack branch refs.
fn symbolic_remote_names_from_seeds(
    repo: &OverlayRepo<'_>,
    seeds: &[Seed],
    project_meta: &ProjectMeta,
    include_seed_refs: bool,
) -> Vec<String> {
    let remote_names = repo.remote_names();
    let refs = seeds
        .iter()
        .filter_map(|seed| {
            include_seed_refs
                .then_some(seed.ref_name.as_ref())
                .flatten()
        })
        .filter_map({
            let remote_names = &remote_names;
            move |ref_name| {
                extract_remote_name_and_short_name(ref_name.as_ref(), remote_names)
                    .map(|(remote, _short_name)| (1, remote))
            }
        });
    let workspace_metadata_names = seeds
        .iter()
        .filter_map(|seed| match seed.metadata.as_ref() {
            Some(SegmentMetadata::Workspace(data)) => Some(data),
            Some(SegmentMetadata::Branch(_)) | None => None,
        })
        .flat_map(|data| {
            data.stacks.iter().flat_map(|s| {
                s.branches.iter().flat_map(|b| {
                    extract_remote_name_and_short_name(b.ref_name.as_ref(), &remote_names)
                        .map(|(remote, _short_name)| (1, remote))
                })
            })
        });
    let desired_refs = seeds.iter().filter_map(|seed| match &seed.role {
        _ if !include_seed_refs => None,
        SeedRole::WorkspaceStackBranch { desired_ref_name } => {
            extract_remote_name_and_short_name(desired_ref_name.as_ref(), &remote_names)
                .map(|(remote, _short_name)| (1, remote))
        }
        SeedRole::Reachable
        | SeedRole::Workspace
        | SeedRole::TargetLocal { .. }
        | SeedRole::TargetRemote => None,
    });
    let target_ref = project_meta.target_ref.as_ref().and_then(|target_ref| {
        extract_remote_name_and_short_name(target_ref.as_ref(), &remote_names)
            .map(|(remote, _short_name)| (1, remote))
    });
    let push_remote = project_meta
        .push_remote
        .as_ref()
        .map(|push_remote| (0, push_remote.clone()));
    sorted_symbolic_remote_names(
        refs.chain(workspace_metadata_names)
            .chain(desired_refs)
            .chain(target_ref)
            .chain(push_remote),
    )
}

/// Sort and deduplicate remote names, preserving explicit push remotes before
/// remotes inferred from refs with the same name.
fn sorted_symbolic_remote_names(names: impl Iterator<Item = (usize, String)>) -> Vec<String> {
    let mut names: Vec<_> = names.collect();
    names.sort();
    names.dedup();
    names.into_iter().map(|(_order, remote)| remote).collect()
}

/// Return whether an initial queue item should be pushed to the front.
///
/// This is the second half of the ordering heuristic. `seeds_in_queue_order()`
/// decides the order in which initial segments are created. Once those segments
/// are converted into traversal queue items, some roles must still be
/// front-loaded so their commits are visited before ordinary reachable or stack
/// branch work that may point at the same commits so they can own them.
///
/// Synthetic integrated seeds are always front-loaded because they represent
/// additional target/limit commits rather than user-visible branch roots. For
/// workspace-related traversals, workspace, integrated target, and target-local
/// seeds are also front-loaded so target ownership and target/local sibling
/// links are established before stack-branch traversal can claim shared commits.
/// Workspace stack branches are deliberately not front-loaded: their segment
/// creation order is recovered from metadata, but their traversal work should
/// follow the workspace/target context.
fn queue_should_frontload_seed(
    seed: &Seed,
    frontload_workspace_related_seeds: bool,
    auxiliary_integrated_seed_ids: &BTreeSet<gix::ObjectId>,
) -> bool {
    seed.is_auxiliary_integrated_seed(auxiliary_integrated_seed_ids)
        || (frontload_workspace_related_seeds
            && matches!(
                seed.role,
                SeedRole::Workspace | SeedRole::TargetRemote | SeedRole::TargetLocal { .. }
            ))
}

/// Return the flags and limit used by a reachable seed seeking the entrypoint.
fn reachable_seed_flags_and_limit(
    seed: gix::ObjectId,
    entrypoint: gix::ObjectId,
    max_limit: Limit,
    goals: &mut Goals,
) -> (CommitFlags, Limit) {
    let limit = if seed == entrypoint {
        max_limit
    } else {
        max_limit.with_indirect_goal(entrypoint, goals)
    };
    (CommitFlags::NotInRemote, limit)
}
