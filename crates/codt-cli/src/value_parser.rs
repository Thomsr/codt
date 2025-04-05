//! clap does not include a value parser for range bounded float parsing. This copies much of the integer `RangedI64ValueParser`, and adapts it to floats.

use std::ops::RangeBounds;

use clap::{
    Arg, Command, Error,
    builder::TypedValueParser,
    error::{ContextKind, ContextValue, ErrorKind},
};

#[derive(Copy, Clone, Debug)]
pub struct RangedF64ValueParser<T: TryFrom<f64> + Clone + Send + Sync = f64> {
    bounds: (std::ops::Bound<f64>, std::ops::Bound<f64>),
    target: std::marker::PhantomData<T>,
}

impl<T: TryFrom<f64> + Clone + Send + Sync> RangedF64ValueParser<T> {
    /// Select full range of `f64`
    pub fn new() -> Self {
        Self::from(..)
    }

    /// Set the supported range
    pub fn range<B: RangeBounds<f64>>(mut self, range: B) -> Self {
        // Consideration: when the user does `value_parser!(u8).range()`
        // - Avoid programming mistakes by accidentally expanding the range
        // - Make it convenient to limit the range like with `..10`
        let start = range.start_bound().cloned();
        let end = range.end_bound().cloned();
        self.bounds = (start, end);
        self
    }
}

impl<T: TryFrom<f64> + Clone + Send + Sync + 'static> TypedValueParser for RangedF64ValueParser<T>
where
    <T as TryFrom<f64>>::Error: Send + Sync + 'static + std::error::Error + ToString,
{
    type Value = T;

    fn parse_ref(
        &self,
        cmd: &Command,
        arg: Option<&Arg>,
        raw_value: &std::ffi::OsStr,
    ) -> Result<Self::Value, Error> {
        let value = raw_value
            .to_str()
            .ok_or_else(|| Error::new(ErrorKind::InvalidUtf8).with_cmd(cmd))?;
        let value = value.parse::<f64>().map_err(|_| {
            let arg = arg
                .map(|a| a.to_string())
                .unwrap_or_else(|| "...".to_owned());

            let mut err = Error::new(ErrorKind::ValueValidation).with_cmd(cmd);
            err.insert(ContextKind::InvalidArg, ContextValue::String(arg));
            err.insert(
                ContextKind::InvalidValue,
                ContextValue::String(raw_value.to_string_lossy().into_owned()),
            );
            err
        })?;
        if !self.bounds.contains(&value) {
            let arg = arg
                .map(|a| a.to_string())
                .unwrap_or_else(|| "...".to_owned());
            let mut err = Error::new(ErrorKind::ValueValidation).with_cmd(cmd);
            err.insert(ContextKind::InvalidArg, ContextValue::String(arg));
            err.insert(
                ContextKind::InvalidValue,
                ContextValue::String(raw_value.to_string_lossy().into_owned()),
            );
            return Err(err);
        }

        let value: Result<Self::Value, _> = value.try_into();
        let value = value.map_err(|_| {
            let arg = arg
                .map(|a| a.to_string())
                .unwrap_or_else(|| "...".to_owned());

            let mut err = Error::new(ErrorKind::ValueValidation).with_cmd(cmd);
            err.insert(ContextKind::InvalidArg, ContextValue::String(arg));
            err.insert(
                ContextKind::InvalidValue,
                ContextValue::String(raw_value.to_string_lossy().into_owned()),
            );
            err
        })?;

        Ok(value)
    }
}

impl<T: TryFrom<f64> + Clone + Send + Sync, B: RangeBounds<f64>> From<B>
    for RangedF64ValueParser<T>
{
    fn from(range: B) -> Self {
        Self {
            bounds: (range.start_bound().cloned(), range.end_bound().cloned()),
            target: Default::default(),
        }
    }
}

impl<T: TryFrom<f64> + Clone + Send + Sync> Default for RangedF64ValueParser<T> {
    fn default() -> Self {
        Self::new()
    }
}
