pub struct Lazy<T> {
    cache: Option<T>,
    func: fn() -> T,
}

impl<T> Lazy<T> {
    pub fn new(func: fn() -> T) -> Self {
        Self {
            func,
            cache: None,
        }
    }

    pub fn value(&mut self) -> &T {
        if self.cache.is_some() {
            self.cache.as_ref().unwrap()
        } else {
            let func = self.func;
            let val = func();
            self.cache = Some(val);
            self.cache.as_ref().unwrap()
        }
    }
}