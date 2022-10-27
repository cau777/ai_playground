/**
 * Asserts the array has a certain length by removing elements from the beginning and return the array.
 * Modifies the original array
 * @param array
 * @param keep
 */
export function keepAtMost<T>(array: T[], keep: number) {
    array.splice(0, Math.max(0, array.length-keep));
    return array;
}