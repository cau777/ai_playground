use crate::chess::decision_tree::DecisionTree;
use crate::chess::decision_tree::node::Node;

const CELL_SIZE: u64 = 50;
const HALF_CELL: u64 = CELL_SIZE / 2;
const CIRCLE_RAD: u64 = 20;

type Positions = Vec<(f64, f64)>;

fn cell_to_pixel(val: f64) -> u64 {
    (val * CELL_SIZE as f64).round() as u64
}

fn find_position(nodes: &[Node], index: usize, result: &mut Positions, current_col: &mut u64) {
    let node = &nodes[index];
    let before_children = *current_col;
    if let Some(children) = &node.children {
        children.iter().for_each(|o| find_position(nodes, *o, result, current_col));
    }
    let after_children = *current_col;
    let delta = after_children - before_children;

    let col = if delta == 0 {
        after_children as f64
    } else if delta % 2 == 0 {
        (after_children + before_children - 1) as f64 * 0.5
    } else {
        (after_children + before_children) as f64 * 0.5
    };
    result[index] = (node.depth as f64, col);
    *current_col += 1;
}

fn create_circles(positions: &Positions) -> String {
    let mut result = String::new();
    for (row, col) in positions {
        result += &format!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"white\"/>",
            cell_to_pixel(*col) + HALF_CELL, cell_to_pixel(*row) + HALF_CELL, CIRCLE_RAD
        );
    }
    result
}

fn create_lines(positions: &Positions, nodes: &[Node]) -> String {
    let mut result = String::new();

    for x in 0..nodes.len() {
        let (xrow, xcol) = positions[x];
        if let Some(children) = &nodes[x].children {
            for c in children {
                let (trow, tcol) = positions[*c];

                result += &format!(
                    "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\"/>",
                    cell_to_pixel(xcol) + HALF_CELL, cell_to_pixel(xrow) + HALF_CELL,
                    cell_to_pixel(tcol) + HALF_CELL, cell_to_pixel(trow) + HALF_CELL
                );
            }
        }
    }

    result
}

fn create_text(positions: &Positions, nodes: &[Node]) -> String {
    let mut result = String::new();

    for x in 0..nodes.len() {
        let (row, col) = positions[x];
        let node = &nodes[x];
        result += &format!(
            "<text x=\"{}\" y=\"{}\">{}-{}</text>",
            cell_to_pixel(col) + HALF_CELL, cell_to_pixel(row-0.15) + HALF_CELL,
            node.movement.from, node.movement.to,
        );
        result += &format!(
            "<text x=\"{}\" y=\"{}\">{:.4}</text>",
            cell_to_pixel(col) + HALF_CELL, cell_to_pixel(row+0.15) + HALF_CELL,
            node.eval
        );
    }

    result
}

impl DecisionTree {
    pub fn to_svg(&self) -> String {
        // let mut ordered_nodes = Vec::with_capacity(self.nodes.len());
        let mut positions = vec![Default::default(); self.nodes.len()];

        let mut current_col = 0;
        find_position(&self.nodes, 0, &mut positions, &mut current_col);
        let max_depth = self.nodes.iter().map(|o| o.depth as u64).max().unwrap();

        format!(
            "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' width='100%' height='100%' viewPort='0 0 {} {}'>\
            <defs><style>line {{ stroke:rgb(255,0,0);stroke-width:5 }} circle {{ stroke:black;stroke-width=4px; }} text {{ text-anchor:middle;font-size:9px;font-family:Verdana,Arial,sans-serif }}</style></defs>\
            {}{}{}</svg>",
            current_col * CELL_SIZE, max_depth * CELL_SIZE,
            create_lines(&positions, &self.nodes),
            create_circles(&positions),
            create_text(&positions, &self.nodes),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::chess::movement::Movement;
    use super::*;

    #[test]
    fn test_to_svg() {
        let mut tree = DecisionTree::new(true);
        tree.submit_node_children(0, &[
            (Movement::from_notations("A2", "A4"), 2.0, Default::default()),
            (Movement::from_notations("B2", "B4"), 1.0, Default::default()),
        ]);

        tree.submit_node_children(1, &[
            (Movement::from_notations("A7", "A5"), -1.0, Default::default()),
            (Movement::from_notations("B7", "B5"), 0.0, Default::default()),
        ]);

        tree.submit_node_children(3, &[
            (Movement::from_notations("C2", "C4"), 2.0, Default::default()),
            (Movement::from_notations("D2", "D4"), 1.0, Default::default()),
        ]);

        tree.submit_node_children(4, &[
            (Movement::from_notations("E2", "E4"), 2.0, Default::default()),
            (Movement::from_notations("F2", "F4"), 1.0, Default::default()),
        ]);

        println!("{}", tree.to_svg());
    }
}