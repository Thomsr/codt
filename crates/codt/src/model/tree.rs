use std::{fmt::Display, sync::Arc};

use crate::tasks::OptimizationTask;

pub enum Tree<OT: OptimizationTask> {
    Branch(BranchNode<OT>),
    Leaf(LeafNode<OT>),
}

impl<OT: OptimizationTask> Display for Tree<OT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Branch(branch) => write!(
                f,
                "[{}<={}:{},{}]",
                branch.split_feature, branch.split_threshold, branch.left_child, branch.right_child
            ),
            Self::Leaf(leaf) => write!(f, "{}", leaf.label),
        }
    }
}

impl<OT: OptimizationTask> Tree<OT> {
    pub fn cost(&self) -> OT::CostType {
        match self {
            Self::Branch(branch) => branch.cost,
            Self::Leaf(leaf) => leaf.cost,
        }
    }

    pub fn predict(&self, features: Vec<f64>) -> OT::LabelType {
        match self {
            Self::Branch(branch) => {
                if features[branch.split_feature] <= branch.split_threshold {
                    branch.left_child.predict(features)
                } else {
                    branch.right_child.predict(features)
                }
            }
            Self::Leaf(leaf) => leaf.label,
        }
    }

    /// Returns an ASCII tree view that is easier to read than the compact bracket format.
    pub fn pretty(&self) -> String {
        fn render<OT: OptimizationTask>(node: &Tree<OT>, prefix: &str, out: &mut String) {
            match node {
                Tree::Leaf(leaf) => {
                    out.push_str(prefix);
                    out.push_str(&format!("leaf: {}\n", leaf.label));
                }
                Tree::Branch(branch) => {
                    out.push_str(prefix);
                    out.push_str(&format!(
                        "[x{} <= {}]\n",
                        branch.split_feature, branch.split_threshold
                    ));

                    out.push_str(prefix);
                    out.push_str("|- yes\n");
                    let left_prefix = format!("{}|  ", prefix);
                    render(branch.left_child.as_ref(), &left_prefix, out);

                    out.push_str(prefix);
                    out.push_str("'- no\n");
                    let right_prefix = format!("{}   ", prefix);
                    render(branch.right_child.as_ref(), &right_prefix, out);
                }
            }
        }

        let mut out = String::new();
        render(self, "", &mut out);
        out
    }
}

pub struct BranchNode<OT: OptimizationTask> {
    pub cost: OT::CostType,
    pub split_feature: usize,
    pub split_threshold: f64,
    pub left_child: Arc<Tree<OT>>,
    pub right_child: Arc<Tree<OT>>,
}

pub struct LeafNode<OT: OptimizationTask> {
    pub cost: OT::CostType,
    pub label: OT::LabelType,
}
