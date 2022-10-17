pub type LazyWithoutArgs<T> = Lazy<T, ()>;

pub trait LazyWithoutArgsOps<T> {
    fn v(&mut self) -> &T;
    fn v_mut(&mut self) -> &mut T;
}

pub struct Lazy<T, TArgs> {
    cache: Option<T>,
    func: fn(TArgs) -> T,
}

impl<T, TArgs> Lazy<T, TArgs> {
    pub fn new(func: fn(TArgs) -> T) -> Self {
        Self { func, cache: None }
    }

    pub fn value(&mut self, args: TArgs) -> &T {
        if self.cache.is_some() {
            self.cache.as_ref().unwrap()
        } else {
            let func = self.func;
            let val = func(args);
            self.cache = Some(val);
            self.cache.as_ref().unwrap()
        }
    }

    pub fn value_mut(&mut self, args: TArgs) -> &mut T {
        if self.cache.is_some() {
            self.cache.as_mut().unwrap()
        } else {
            let func = self.func;
            let val = func(args);
            self.cache = Some(val);
            self.cache.as_mut().unwrap()
        }
    }
}

impl<T> LazyWithoutArgsOps<T> for Lazy<T, ()> {
    fn v(&mut self) -> &T {
        Self::value(self, ())
    }

    fn v_mut(&mut self) -> &mut T {
        Self::value_mut(self, ())
    }
}
