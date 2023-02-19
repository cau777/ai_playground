use std::fs::File;
use std::io;
use std::io::{BufRead, BufReader};
use crate::chess::movement::Movement;

#[derive(Debug)]
/// All openings represented in a tree-like structure where each node is a movement that can have a name or be part of a variation
/// This allows easily finding continuations for an opening move, and getting the current opening name at any point
pub struct OpeningsTree {
    nodes: Vec<OpeningNode>,
}

#[derive(Debug)]
struct OpeningNode {
    name: String,
    movement: Movement,
    parent: usize,
    children: Vec<usize>,
}

impl OpeningsTree {
    /// Build an opening tree based on a file where each line is in the format
    /// opening_name|from_to|connections_indexes
    /// Example: A06 Reti Opening|d7d5|7,10
    pub fn load_from_file(file: File) -> io::Result<Self> {
        let reader = BufReader::new(file);
        Self::load(reader.lines())
    }

    /// Build an opening tree based on a string where each line is in the format
    /// opening_name|from_to|connections_indexes
    /// Example: A06 Reti Opening|d7d5|7,10
    pub fn load_from_string(string: &str) -> io::Result<Self> {
        Self::load(string.split('\n').map(|o| Ok(o.to_owned())))
    }

    fn load(iter: impl Iterator<Item=io::Result<String>>) -> io::Result<Self> {
        let mut nodes = Vec::new();
        let mut first = true;

        for line in iter {
            let line = line?;
            let mut split = line.split('|');
            let name = split.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing line part"))?;
            let notation = split.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing line part"))?;
            let connections = split.next().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing line part"))?;

            let movement = if first {
                Movement::from_notations("A1", "A1") // This value will never be used
            } else {
                Movement::try_from_notations(&notation[..2], &notation[2..])
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid notation"))?
            };

            nodes.push(OpeningNode {
                name: name.to_owned(),
                children: connections.split(',').filter_map(|o| o.parse().ok()).collect(),
                movement,
                parent: usize::MAX,
            });
            first = false;
        }

        for i in 0..nodes.len() {
            for child in nodes[i].children.clone() {
                nodes[child].parent = i;
            }
        }

        Ok(Self { nodes })
    }

    // fn set_total_children(nodes: &mut [OpeningNode], index: usize) -> usize {
    //     let node = &nodes[index];
    //
    //     if node.children.is_empty() {
    //         1
    //     } else {
    //         let mut result = 0;
    //         for &c in &node.children {
    //             result += Self::set_total_children(nodes, c);
    //         }
    //         nodes[index].total_children_count = result;
    //         result
    //     }
    // }

    pub fn get_opening_name(&self, node_index: usize) -> &str {
        let mut option = self.nodes.get(node_index);
        while let Some(node) = option {
            if !node.name.is_empty() {
                return &node.name;
            }
            option = self.nodes.get(node.parent);
        }
        ""
    }

    pub fn get_opening_continuations(&self, node_index: usize) -> Vec<Movement> {
        let node = &self.nodes[node_index];
        node.children.iter()
            .map(|o| self.nodes[*o].movement)
            .collect()
    }

    pub fn find_opening_move(&self, current: usize, movement: Movement) -> Option<usize> {
        let node = &self.nodes[current];
        for child in &node.children {
            if self.nodes[*child].movement == movement {
                return Some(*child);
            }
        }
        None
    }
}