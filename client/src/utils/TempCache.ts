export class TempCache<T> {
    private readonly map = new Map<string, T>();
    
    public get_or(id: string, or: () => T) {
        let cached = this.map.get(id);
        if (cached === undefined) {
            cached = or();
            this.map.set(id, cached);
        }
        return cached;
    }
}