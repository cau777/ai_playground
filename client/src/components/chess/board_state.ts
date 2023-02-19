export type BoardState = {
    board: string;
    possible: Map<string, Set<string>>;
    gameId: string;
    state: string;
    opening: string;
    initialSide: boolean;
}

export function createMapSet(possible: [string, string][]) {
    let result = new Map<string, Set<string>>();
    
    for (let [from, to] of possible) {
        if (!result.has(from))
            result.set(from, new Set<string>());
        
        let set = result.get(from)!;
        set.add(to);
    }
    
    return result;
}