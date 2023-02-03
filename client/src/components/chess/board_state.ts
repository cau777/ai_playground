export type BoardState = {
    board: string;
    possible: Map<string, Set<string>>;
    gameId: string;
    state: string;
    opening: string;
}