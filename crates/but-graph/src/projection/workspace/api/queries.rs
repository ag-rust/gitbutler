//! Discoverable queries over [`Workspace`](crate::Workspace).
//!
//! These functions name the question being asked instead of exposing legacy
//! presentation shapes.

use anyhow::Context;

use crate::{Direction, Workspace, segment, workspace::TargetRef};

/// Legacy query helpers kept for callers that still depend on compatibility
/// semantics.
#[cfg(feature = "legacy")]
#[path = "legacy.rs"]
pub mod legacy;

/// # Points of Interest
impl Workspace {
    /// The name of the segment owning the workspace's lower bound — the ref marking the
    /// common base all stacks converge on (e.g. the target's local `main`), if the bound
    /// exists and its segment is named.
    pub fn lower_bound_ref_name(&self) -> Option<&gix::refs::FullNameRef> {
        self.graph[self.lower_bound_segment_id?].ref_name()
    }

    /// Return the `commit` at the tip of the workspace, or that the tip reference
    /// was pointing to in Git.
    ///
    /// Empty virtual workspace tip segments may fan out to multiple stack
    /// branches, so the workspace segment has no unique graph path to a commit.
    /// This falls back to the peeled commit id stored in the workspace segment's
    /// [`crate::RefInfo`] and resolves that id against the final graph.
    ///
    /// Note that this commit could also be the base of the workspace,
    /// particularly if there are no commits in the workspace.
    pub fn tip_commit(&self) -> Option<&segment::Commit> {
        self.tip_commit_by_segment_id(self.id)
    }

    /// Return the `commit` at the tip of `segment_id`, or that its ref was pointing
    /// to in Git.
    ///
    /// This first uses [`Graph::tip_skip_empty()`](crate::Graph::tip_skip_empty)
    /// to follow an unambiguous chain of empty segments to the first commit.
    /// If that cannot resolve a commit, it falls back to the peeled commit id
    /// stored in the segment's [`crate::RefInfo`] and resolves that id in the
    /// graph.
    ///
    /// That fallback is what makes this useful for workspace-owned virtual
    /// segments whose ref points at a commit, but whose graph edges do not form
    /// a single unambiguous path to it.
    pub fn tip_commit_by_segment_id(&self, segment_id: usize) -> Option<&segment::Commit> {
        self.graph.tip_skip_empty(segment_id).or_else(|| {
            let commit_id = self.graph[segment_id].ref_info.as_ref()?.commit_id?;
            self.graph
                .segment_by_commit_id(commit_id)
                .ok()?
                .commit_by_id(commit_id)
        })
    }

    /// Return the id of the first commit reachable from `segment_id` through an
    /// unambiguous chain of empty segments.
    ///
    /// Unlike [`Self::tip_commit_by_segment_id()`], this does *not* fall back to the
    /// commit id stored in the segment's ref-info — callers that treat "no commit on
    /// the graph line" differently from "no commit at all" need this stricter form.
    pub fn tip_commit_id_skip_empty(&self, segment_id: usize) -> Option<gix::ObjectId> {
        self.graph
            .tip_skip_empty(segment_id)
            .map(|commit| commit.id)
    }

    /// Return `(segment_id, commit_id)` for the segment that is either named `name`, or
    /// that has a commit carrying `name` in its refs. The commit is the one `name` points
    /// to, directly or indirectly.
    ///
    /// Note that tags may or may not be included in the graph, depending on how it was created.
    ///
    /// ### Performance
    ///
    /// This is a brute-force search through all segments - beware of hot-loop usage.
    pub fn segment_id_and_commit_id_by_ref_name(
        &self,
        name: &gix::refs::FullNameRef,
    ) -> Option<(usize, gix::ObjectId)> {
        self.graph
            .segment_and_commit_by_ref_name(name)
            .map(|(segment, commit)| (segment.id, commit.id))
    }

    /// Return the id of the segment owning `commit_id`, or an error if no segment
    /// contains that commit.
    pub fn segment_id_by_commit_id(&self, commit_id: gix::ObjectId) -> anyhow::Result<usize> {
        Ok(self.graph.segment_by_commit_id(commit_id)?.id)
    }

    /// Return the id of the segment acting as merge-base between the segments `a` and `b`,
    /// or `None` if they are disjoint in the commit-graph.
    pub fn merge_base_segment_id(&self, a: usize, b: usize) -> Option<usize> {
        self.graph.find_merge_base(a, b)
    }

    /// Return the full [`RefInfo`](crate::RefInfo) of the workspace tip segment, if the tip
    /// is named. Unlike [`Self::ref_name()`](Self::ref_name) this includes worktree and
    /// pointed-to-commit information.
    pub fn tip_ref_info(&self) -> Option<&crate::RefInfo> {
        self.graph[self.id].ref_info.as_ref()
    }

    /// Return the id of the commit the traversal entrypoint sits on — usually where
    /// `HEAD` points — or `None` if the entrypoint segment carries no commit.
    pub fn entrypoint_commit_id(&self) -> anyhow::Result<Option<gix::ObjectId>> {
        Ok(self.graph.entrypoint()?.commit().map(|commit| commit.id))
    }

    /// Return `true` if the traversal entrypoint is the workspace tip segment itself.
    ///
    /// Note the difference to [`Self::is_entrypoint()`](Self::is_entrypoint): that one asks
    /// whether no stack segment holds the entrypoint, which is also `true` when the
    /// entrypoint sits outside the workspace entirely.
    pub fn tip_is_entrypoint(&self) -> anyhow::Result<bool> {
        Ok(self.graph.entrypoint()?.segment.id == self.id)
    }

    /// Return the stored target commit id.
    ///
    /// This is the previous target position remembered in workspace metadata.
    /// It is normally the base the workspace last integrated with, and
    /// intentionally differs from the current tip of the target reference.
    pub fn stored_target_commit_id(&self) -> Option<gix::ObjectId> {
        self.target_commit.as_ref().map(|target| target.commit_id)
    }

    /// Return the commit id that currently acts as the workspace target.
    ///
    /// This follows the same precedence as operations that need a concrete
    /// target side: target ref tip, then stored target commit, then the first
    /// integrated traversal tip.
    pub fn effective_target_commit_id(&self) -> Option<gix::ObjectId> {
        self.target_ref
            .as_ref()
            .and_then(|target| self.tip_commit_by_segment_id(target.segment_index))
            .map(|commit| commit.id)
            .or_else(|| self.target_commit.as_ref().map(|target| target.commit_id))
            .or_else(|| {
                self.graph
                    .integrated_tip_segments()
                    .into_iter()
                    .find_map(|segment_index| {
                        self.tip_commit_by_segment_id(segment_index)
                            .map(|commit| commit.id)
                    })
            })
    }

    /// Return the segment that currently acts as the workspace target.
    ///
    /// This follows target ref, then stored target commit, then the first
    /// integrated traversal seed in that order.
    pub(crate) fn effective_target_segment_index(&self) -> Option<usize> {
        self.target_ref
            .as_ref()
            .map(|target| target.segment_index)
            .or(self
                .target_commit
                .as_ref()
                .map(|target| target.segment_index))
            .or_else(|| self.graph.integrated_tip_segments().into_iter().next())
    }
}

/// # Carried Data
///
/// Data assembled during graph construction and carried into the projection.
impl Workspace {
    /// The commit graph the segment graph was assembled from — the commit-addressed
    /// substrate. Empty only for hand-assembled graphs that never went through the builders.
    pub fn commit_graph(&self) -> &crate::CommitGraph {
        self.graph.commit_graph()
    }

    /// The project metadata as it was resolved when the graph was built.
    pub fn project_meta(&self) -> &but_core::ref_metadata::ProjectMeta {
        &self.graph.project_meta
    }

    /// Return `true` if the graph traversal was stopped early due to a hard limit,
    /// meaning the graph — and thus this projection — may be incomplete.
    pub fn hard_limit_hit(&self) -> bool {
        self.graph.hard_limit_hit()
    }

    /// The LOCAL branch tracking `remote`, as the builder derived it from the repository
    /// (a git-configured binding, or name-deduction against the workspace's symbolic remotes).
    /// `None` when no local tracks `remote`, or for graphs not born from the builders.
    pub fn local_tracking_branch(
        &self,
        remote: &gix::refs::FullNameRef,
    ) -> Option<&gix::refs::FullName> {
        self.graph.local_tracking_branch(remote)
    }
}

/// # Refs of Interest
impl Workspace {
    /// Return the configured target reference name if the workspace target was
    /// resolved to a branch during graph traversal.
    /// This is mere convenience and it should only be used for displaying the target ref.
    /// For everything else, use [`Self::target_ref`].
    pub fn target_ref_name(&self) -> Option<&gix::refs::FullNameRef> {
        self.target_ref
            .as_ref()
            .map(|target| target.ref_name.as_ref())
    }
}

/// # Sets of Interest
impl Workspace {
    /// Visit the commits in the ancestry of the workspace tip segment — excluding that
    /// segment's own commits — in traversal order, stopping segment-wise at the workspace
    /// [lower bound](Self::lower_bound). Return `true` from `visit` to stop the traversal.
    ///
    /// This is useful to search the workspace history, e.g. for a workspace commit buried
    /// beneath commits made by hand.
    pub fn visit_commits_below_tip(&self, mut visit: impl FnMut(&segment::Commit) -> bool) {
        // The depth safety-bound, expressed over the carried commit graph: a segment is beyond
        // the lower bound when its first commit sits below the bound commit (cg generation:
        // tips high, roots low).
        let cg_bound =
            self.lower_bound_segment_id
                .and_then(|sidx| -> Option<(u32, &crate::CommitGraph)> {
                    let cg = self.graph.commit_graph();
                    let lb = self.graph[sidx].commits.first()?.id;
                    Some((cg.node(lb)?.generation, cg))
                });

        let mut stopped = false;
        self.graph
            .visit_all_segments_excluding_start_until(self.id, Direction::Outgoing, |s| {
                let beyond_bound = cg_bound.is_some_and(|(bound, cg)| {
                    s.commits
                        .first()
                        .and_then(|c| cg.node(c.id))
                        .is_some_and(|n| n.generation < bound)
                });
                if stopped || beyond_bound {
                    return true;
                }
                for commit in &s.commits {
                    if visit(commit) {
                        stopped = true;
                        return true;
                    }
                }
                false
            });
    }

    /// Return all target-reference commits that are ahead of the workspace base,
    /// which is the commits counted with
    /// [workspace::TargetRef::commits_ahead](crate::workspace::TargetRef::commits_ahead)
    ///
    /// The traversal starts at the resolved target reference and stops at the
    /// workspace lower bound or at commits already marked as belonging to the
    /// workspace. The result is ordered in graph traversal order from newer
    /// commits toward older commits.
    pub fn incoming_target_commit_ids(&self) -> anyhow::Result<Vec<gix::ObjectId>> {
        let target_ref = self
            .target_ref
            .as_ref()
            .context("incoming target commits require a workspace with a target ref")?;
        let lower_bound = self.lower_bound_segment_id;

        let mut commit_ids = Vec::new();
        TargetRef::visit_upstream_commits(
            &self.graph,
            target_ref.segment_index,
            lower_bound,
            |segment| {
                commit_ids.extend(segment.commits.iter().map(|commit| commit.id));
            },
        );
        Ok(commit_ids)
    }

    /// Return the ids of all target-side commits that are not part of the workspace's
    /// shared history, in traversal order from the target tip downward.
    /// Return `None` if the workspace has no notion of a target at all.
    ///
    /// The target side is the target ref if resolved, otherwise the stored target commit.
    ///
    /// RULING (2026-07-04): a DISJOINT target contributes no upstream commits. One pass
    /// decides everything on the carried commit graph: paint the lower bound's ancestor set
    /// (= shared history, following connected edges only), then walk down from the target
    /// tip collecting commits until the walk TOUCHES shared history. Diverged commits are
    /// never in that set, so a rewritten remote is collected at any depth; if the walk never
    /// touches it, target and workspace share no history — as far as the graph can see —
    /// and nothing is upstream.
    pub fn upstream_commit_ids_outside_shared_history(&self) -> Option<Vec<gix::ObjectId>> {
        let target_tip = self
            .target_ref
            .as_ref()
            .map(|t| t.segment_index)
            .or(self.target_commit.as_ref().map(|t| t.segment_index))?;
        let shared_history = self.lower_bound.and_then(|lb| {
            let cg = self.graph.commit_graph();
            // An empty (default) commit graph — hand-assembled graphs — knows no ancestry.
            cg.node(lb).is_some().then(|| cg.ancestor_set(lb))
        });
        let mut upstream_commits = Vec::new();
        let mut touched_shared_history = false;
        self.graph
            .visit_all_segments_including_start_until(target_tip, Direction::Outgoing, |s| {
                let prune = true;
                let in_shared_history = shared_history.as_ref().is_some_and(|shared| {
                    s.commits.first().is_some_and(|c| shared.contains(&c.id))
                });
                touched_shared_history |= in_shared_history;
                let is_lower_bound = s
                    .commits
                    .first()
                    .is_some_and(|c| Some(c.id) == self.lower_bound);
                if is_lower_bound || in_shared_history {
                    return prune;
                }
                for c in &s.commits {
                    upstream_commits.push(c.id);
                }
                !prune
            });
        if shared_history.is_some() && !touched_shared_history {
            // Never touching shared history means either a truly DISJOINT target (nothing is
            // upstream) or a target whose connection point lies beyond the traversal window
            // (a diverged remote cut short by limits — its commits ARE upstream). The graph
            // records where the traversal cut: if every collected commit's ancestry is fully
            // present and roots out without meeting shared history, it is genuinely disjoint.
            let cg = self.graph.commit_graph();
            let genuinely_disjoint = !upstream_commits.iter().any(|id| cg.has_cut_parents(*id))
                && !shared_history
                    .iter()
                    .flatten()
                    .any(|id| cg.has_cut_parents(*id));
            if genuinely_disjoint {
                upstream_commits.clear();
            }
        }
        Some(upstream_commits)
    }

    /// Return a [`StackTip`] for each segment strictly below the
    /// reference `name` that coincides with a workspace stack tip — matched by segment id,
    /// or by first commit id to also catch the same commit in a differently segmented spot.
    /// The traversal prunes below each match.
    ///
    /// Return `None` if `name` isn't in the graph at all, which callers may not consider fatal.
    ///
    /// This tells which stacks a to-be-applied branch already flows into, i.e. which stacks
    /// it would supersede.
    pub fn stack_tip_segments_below_ref(
        &self,
        name: &gix::refs::FullNameRef,
    ) -> Option<Vec<StackTip>> {
        let start = self.graph.segment_by_ref_name(name)?.id;
        let stack_tips: Vec<_> = self
            .stacks
            .iter()
            .filter_map(|stack| {
                stack
                    .segments
                    .first()
                    .and_then(|s| s.commits.first().map(|c| (c.id, s.id)))
            })
            .collect();
        let mut matches = Vec::new();
        self.graph.visit_all_segments_excluding_start_until(
            start,
            Direction::Outgoing,
            |segment| {
                let matched = stack_tips.iter().any(|(cid, sidx)| {
                    segment.id == *sidx || segment.commits.first().is_some_and(|c| c.id == *cid)
                });
                if matched {
                    matches.push(StackTip {
                        segment_id: segment.id,
                        ref_name: segment.ref_name().map(|rn| rn.to_owned()),
                        commit_id: segment.commits.first().map(|c| c.id),
                    });
                }
                matched
            },
        );
        Some(matches)
    }
}

/// A workspace stack tip in plain data, as returned by
/// [`Workspace::stack_tip_segments_below_ref()`].
#[derive(Debug, Clone)]
pub struct StackTip {
    /// The id of the segment holding the tip.
    pub segment_id: usize,
    /// The name of the tip segment, if it is named.
    pub ref_name: Option<gix::refs::FullName>,
    /// The tip segment's first commit, absent for empty segments.
    pub commit_id: Option<gix::ObjectId>,
}

/// # Predicates
impl Workspace {
    /// Return `true` if `commit_id` occurs on the first-parent branch line of
    /// `start_segment_id`, including that segment's own commits.
    ///
    /// This is stricter than an all-parents reachability test on purpose: a merge can
    /// make a commit reachable without making it part of the branch's own line, and
    /// callers reasoning about pushability need the branch line, not reachability.
    pub fn first_parent_line_contains_commit(
        &self,
        start_segment_id: usize,
        commit_id: gix::ObjectId,
    ) -> bool {
        if self.graph[start_segment_id]
            .commits
            .iter()
            .any(|commit| commit.id == commit_id)
        {
            return true;
        }
        let mut found = false;
        self.graph
            .visit_segments_downward_along_first_parent_exclude_start(
                start_segment_id,
                |segment| {
                    found = segment.commits.iter().any(|commit| commit.id == commit_id);
                    found
                },
            );
        found
    }
}
