//! Exact compressed token index. This contains no model state or GPU addresses.
use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Default)]
struct Node { edge: Vec<u32>, children: BTreeMap<u32, Box<Node>> }
#[derive(Default)]
pub struct Tree { root: Node }
impl Node {
    fn observe(&mut self, seq: &[u32]) -> usize {
        let mut node = self;
        let mut offset = 0;
        while offset < seq.len() {
            let first = seq[offset];
            if !node.children.contains_key(&first) {
                node.children.insert(first, Box::new(Node {
                    edge: seq[offset..].to_vec(), children: BTreeMap::new()
                }));
                return offset;
            }
            let child = node.children.get_mut(&first).unwrap();
            let shared = child.edge.iter().zip(&seq[offset..])
                .take_while(|(a, b)| a == b).count();
            if shared == child.edge.len() {
                offset += shared; node = child; continue;
            }
            let suffix = child.edge.split_off(shared);
            let old_children = std::mem::take(&mut child.children);
            child.children.insert(suffix[0], Box::new(Node { edge: suffix, children: old_children }));
            offset += shared;
            if offset < seq.len() {
                child.children.insert(seq[offset], Box::new(Node {
                    edge: seq[offset..].to_vec(), children: BTreeMap::new()
                }));
            }
            return offset;
        }
        offset
    }
    fn matched(&self, seq: &[u32]) -> usize {
        let mut node = self; let mut offset = 0;
        while offset < seq.len() {
            let child = match node.children.get(&seq[offset]) { Some(v) => v, None => break };
            let shared = child.edge.iter().zip(&seq[offset..])
                .take_while(|(a, b)| a == b).count();
            offset += shared;
            if shared != child.edge.len() { break; }
            node = child;
        }
        offset
    }
}
#[no_mangle]
pub extern "C" fn jev_radix_new() -> *mut Tree { Box::into_raw(Box::new(Tree::default())) }
#[no_mangle]
pub unsafe extern "C" fn jev_radix_free(tree: *mut Tree) {
    if !tree.is_null() { drop(Box::from_raw(tree)); }
}
#[no_mangle]
pub unsafe extern "C" fn jev_radix_observe(tree: *mut Tree, ids: *const u32, len: usize) -> usize {
    if tree.is_null() || (ids.is_null() && len != 0) { return usize::MAX; }
    let seq = if len == 0 { &[] } else { std::slice::from_raw_parts(ids, len) };
    catch_unwind(AssertUnwindSafe(|| (*tree).root.observe(seq))).unwrap_or(usize::MAX)
}
#[no_mangle]
pub unsafe extern "C" fn jev_radix_match(tree: *const Tree, ids: *const u32, len: usize) -> usize {
    if tree.is_null() || (ids.is_null() && len != 0) { return usize::MAX; }
    let seq = if len == 0 { &[] } else { std::slice::from_raw_parts(ids, len) };
    catch_unwind(AssertUnwindSafe(|| (*tree).root.matched(seq))).unwrap_or(usize::MAX)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn exact_split_and_query() {
        let mut t = Tree::default();
        assert_eq!(t.root.observe(&[1,2,3,4]), 0);
        assert_eq!(t.root.observe(&[1,2,7]), 2);
        assert_eq!(t.root.matched(&[1,2,3,9]), 3);
        assert_eq!(t.root.matched(&[1,2,8]), 2);
        assert_eq!(t.root.matched(&[9]), 0);
        assert_eq!(t.root.observe(&[1,2]), 2);
        assert_eq!(t.root.observe(&[1,2,7]), 3);
    }
    #[test] fn compare_with_longest_prefix_oracle() {
        let mut t=Tree::default(); let mut history: Vec<Vec<u32>>=vec![];
        let mut seed=7u32;
        for _ in 0..2000 {
            seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let len=(seed%32+1) as usize; let mut seq=Vec::new();
            for _ in 0..len { seed=seed.wrapping_mul(1664525).wrapping_add(1013904223); seq.push((seed>>16)%8); }
            let expected=history.iter().map(|h|h.iter().zip(&seq).take_while(|(a,b)|a==b).count()).max().unwrap_or(0);
            assert_eq!(t.root.matched(&seq),expected);
            assert_eq!(t.root.observe(&seq),expected);
            history.push(seq);
        }
    }
}
