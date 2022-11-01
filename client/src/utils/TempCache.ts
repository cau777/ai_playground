// TODO
export class TempCache<T> {
    public get_or(id: string, or: () => T) {
        return or();
    }
}