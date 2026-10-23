//! O(3) irreps and explicit contiguous feature layout metadata.

use std::{fmt, str::FromStr};

use crate::{CoreError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parity {
    Even,
    Odd,
}

impl Parity {
    #[must_use]
    pub fn from_char(value: char) -> Option<Self> {
        match value {
            'e' => Some(Self::Even),
            'o' => Some(Self::Odd),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_char(self) -> char {
        match self {
            Self::Even => 'e',
            Self::Odd => 'o',
        }
    }

    #[must_use]
    pub fn multiply(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Even, Self::Even) | (Self::Odd, Self::Odd) => Self::Even,
            (Self::Even, Self::Odd) | (Self::Odd, Self::Even) => Self::Odd,
        }
    }
}

impl fmt::Display for Parity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Irrep {
    pub l: usize,
    pub parity: Parity,
}

impl Irrep {
    pub fn new(l: usize, parity: Parity) -> Result<Self> {
        let irrep = Self { l, parity };
        irrep.checked_dim()?;
        Ok(irrep)
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.checked_dim().unwrap_or(usize::MAX)
    }

    pub fn checked_dim(&self) -> Result<usize> {
        self.l
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid_irreps(format!("l={} overflows irrep dimension", self.l)))
    }
}

impl FromStr for Irrep {
    type Err = CoreError;

    fn from_str(input: &str) -> Result<Self> {
        let text = input.trim();
        if text.is_empty() {
            return Err(invalid_irreps("empty irrep"));
        }

        let parity_char = text
            .chars()
            .last()
            .ok_or_else(|| invalid_irreps("empty irrep"))?;
        let parity = Parity::from_char(parity_char)
            .ok_or_else(|| invalid_irreps(format!("unknown parity '{parity_char}' in '{text}'")))?;
        let l_text = &text[..text.len() - parity_char.len_utf8()];

        if l_text.is_empty() {
            return Err(invalid_irreps(format!("missing l in '{text}'")));
        }
        if !l_text.chars().all(|value| value.is_ascii_digit()) {
            return Err(invalid_irreps(format!("invalid l '{l_text}' in '{text}'")));
        }

        let l = l_text
            .parse::<usize>()
            .map_err(|_| invalid_irreps(format!("invalid l '{l_text}' in '{text}'")))?;
        Self::new(l, parity)
    }
}

impl fmt::Display for Irrep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.l, self.parity)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IrrepMul {
    pub mul: usize,
    pub irrep: Irrep,
}

impl IrrepMul {
    pub fn new(mul: usize, irrep: Irrep) -> Result<Self> {
        if mul == 0 {
            return Err(invalid_irreps("multiplicity must be greater than zero"));
        }

        let value = Self { mul, irrep };
        value.checked_dim()?;
        Ok(value)
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.checked_dim().unwrap_or(usize::MAX)
    }

    pub fn checked_dim(&self) -> Result<usize> {
        self.irrep
            .checked_dim()?
            .checked_mul(self.mul)
            .ok_or_else(|| {
                invalid_irreps(format!("{}x{} overflows dimension", self.mul, self.irrep))
            })
    }
}

impl fmt::Display for IrrepMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.mul, self.irrep)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Irreps {
    terms: Vec<IrrepMul>,
}

impl Irreps {
    pub fn new(terms: Vec<IrrepMul>) -> Result<Self> {
        if terms.is_empty() {
            return Err(invalid_irreps("irreps must contain at least one term"));
        }

        let irreps = Self { terms };
        irreps.checked_dim()?;
        Ok(irreps)
    }

    #[must_use]
    pub fn terms(&self) -> &[IrrepMul] {
        &self.terms
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.checked_dim().unwrap_or(usize::MAX)
    }

    pub fn checked_dim(&self) -> Result<usize> {
        self.terms.iter().try_fold(0usize, |total, term| {
            total
                .checked_add(term.checked_dim()?)
                .ok_or_else(|| invalid_irreps("irreps total dimension overflow"))
        })
    }

    #[must_use]
    pub fn layout(&self) -> Vec<LayoutSegment> {
        self.checked_layout().unwrap_or_default()
    }

    pub fn checked_layout(&self) -> Result<Vec<LayoutSegment>> {
        let mut start = 0usize;
        let capacity = self.terms.iter().try_fold(0usize, |total, term| {
            total
                .checked_add(term.mul)
                .ok_or_else(|| invalid_irreps("irreps layout segment count overflow"))
        })?;
        let mut segments = Vec::with_capacity(capacity);

        for (term_index, term) in self.terms.iter().enumerate() {
            let len = term.irrep.checked_dim()?;
            for mul_index in 0..term.mul {
                segments.push(LayoutSegment {
                    term_index,
                    mul_index,
                    irrep: term.irrep,
                    start,
                    len,
                });
                start = start
                    .checked_add(len)
                    .ok_or_else(|| invalid_irreps("irreps layout offset overflow"))?;
            }
        }

        Ok(segments)
    }
}

impl FromStr for Irreps {
    type Err = CoreError;

    fn from_str(input: &str) -> Result<Self> {
        let text = input.trim();
        if text.is_empty() {
            return Err(invalid_irreps("empty irreps string"));
        }

        let mut terms = Vec::new();
        for (term_index, raw_term) in text.split('+').enumerate() {
            let term = raw_term.trim();
            if term.is_empty() {
                return Err(invalid_irreps(format!(
                    "empty term at position {term_index} in '{text}'"
                )));
            }
            terms.push(parse_irrep_mul(term)?);
        }

        Self::new(terms)
    }
}

impl fmt::Display for Irreps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, term) in self.terms.iter().enumerate() {
            if index > 0 {
                f.write_str(" + ")?;
            }
            write!(f, "{term}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutSegment {
    pub term_index: usize,
    pub mul_index: usize,
    pub irrep: Irrep,
    pub start: usize,
    pub len: usize,
}

impl LayoutSegment {
    #[must_use]
    pub fn end(&self) -> usize {
        self.start.saturating_add(self.len)
    }
}

fn parse_irrep_mul(input: &str) -> Result<IrrepMul> {
    if input.chars().any(char::is_whitespace) {
        return Err(invalid_irreps(format!("whitespace inside term '{input}'")));
    }

    if let Some((mul_text, irrep_text)) = input.split_once('x') {
        if irrep_text.contains('x') {
            return Err(invalid_irreps(format!(
                "too many 'x' separators in '{input}'"
            )));
        }
        if mul_text.is_empty() || irrep_text.is_empty() {
            return Err(invalid_irreps(format!(
                "malformed multiplicity term '{input}'"
            )));
        }
        if !mul_text.chars().all(|value| value.is_ascii_digit()) {
            return Err(invalid_irreps(format!(
                "invalid multiplicity '{mul_text}' in '{input}'"
            )));
        }

        let mul = mul_text.parse::<usize>().map_err(|_| {
            invalid_irreps(format!("invalid multiplicity '{mul_text}' in '{input}'"))
        })?;
        let irrep = irrep_text.parse::<Irrep>()?;
        IrrepMul::new(mul, irrep)
    } else {
        IrrepMul::new(1, input.parse::<Irrep>()?)
    }
}

fn invalid_irreps(message: impl Into<String>) -> CoreError {
    CoreError::InvalidIrreps(message.into())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn parses_irrep() {
        assert_eq!(
            "2e".parse::<Irrep>().unwrap(),
            Irrep {
                l: 2,
                parity: Parity::Even,
            }
        );
        assert_eq!(
            "1o".parse::<Irrep>().unwrap(),
            Irrep {
                l: 1,
                parity: Parity::Odd,
            }
        );
    }

    #[test]
    fn parses_irreps_with_multiplicities() {
        let irreps = "1x0e + 2x1o + 4x2e".parse::<Irreps>().unwrap();

        assert_eq!(
            irreps.terms(),
            &[
                IrrepMul {
                    mul: 1,
                    irrep: Irrep {
                        l: 0,
                        parity: Parity::Even,
                    },
                },
                IrrepMul {
                    mul: 2,
                    irrep: Irrep {
                        l: 1,
                        parity: Parity::Odd,
                    },
                },
                IrrepMul {
                    mul: 4,
                    irrep: Irrep {
                        l: 2,
                        parity: Parity::Even,
                    },
                },
            ]
        );
        assert_eq!(irreps.to_string(), "1x0e + 2x1o + 4x2e");
    }

    #[test]
    fn parses_irreps_with_implicit_unit_multiplicity() {
        let irreps = "0e + 1o".parse::<Irreps>().unwrap();

        assert_eq!(
            irreps.terms(),
            &[
                IrrepMul {
                    mul: 1,
                    irrep: Irrep {
                        l: 0,
                        parity: Parity::Even,
                    },
                },
                IrrepMul {
                    mul: 1,
                    irrep: Irrep {
                        l: 1,
                        parity: Parity::Odd,
                    },
                },
            ]
        );
        assert_eq!(irreps.to_string(), "1x0e + 1x1o");
    }

    #[test]
    fn computes_dimensions() {
        let irreps = "1x0e + 2x1o + 4x2e".parse::<Irreps>().unwrap();

        assert_eq!(Irrep::new(0, Parity::Even).unwrap().dim(), 1);
        assert_eq!(Irrep::new(1, Parity::Odd).unwrap().dim(), 3);
        assert_eq!(Irrep::new(2, Parity::Even).unwrap().dim(), 5);
        assert_eq!(
            IrrepMul::new(4, Irrep::new(2, Parity::Even).unwrap())
                .unwrap()
                .dim(),
            20
        );
        assert_eq!(irreps.dim(), 27);
    }

    #[test]
    fn creates_contiguous_layout_segments_per_multiplicity() {
        let irreps = "1x0e + 2x1o + 1x2e".parse::<Irreps>().unwrap();

        assert_eq!(
            irreps.layout(),
            vec![
                LayoutSegment {
                    term_index: 0,
                    mul_index: 0,
                    irrep: Irrep {
                        l: 0,
                        parity: Parity::Even,
                    },
                    start: 0,
                    len: 1,
                },
                LayoutSegment {
                    term_index: 1,
                    mul_index: 0,
                    irrep: Irrep {
                        l: 1,
                        parity: Parity::Odd,
                    },
                    start: 1,
                    len: 3,
                },
                LayoutSegment {
                    term_index: 1,
                    mul_index: 1,
                    irrep: Irrep {
                        l: 1,
                        parity: Parity::Odd,
                    },
                    start: 4,
                    len: 3,
                },
                LayoutSegment {
                    term_index: 2,
                    mul_index: 0,
                    irrep: Irrep {
                        l: 2,
                        parity: Parity::Even,
                    },
                    start: 7,
                    len: 5,
                },
            ]
        );
        assert_eq!(irreps.layout().last().unwrap().end(), irreps.dim());
    }

    #[test]
    fn public_irrep_fields_do_not_panic_on_overflow() {
        let irrep = Irrep {
            l: usize::MAX,
            parity: Parity::Even,
        };
        let mul = IrrepMul { mul: 2, irrep };
        let irreps = Irreps { terms: vec![mul] };

        assert_eq!(irrep.dim(), usize::MAX);
        assert_eq!(mul.dim(), usize::MAX);
        assert_eq!(irreps.dim(), usize::MAX);
        assert!(irrep.checked_dim().is_err());
        assert!(irreps.checked_layout().is_err());
        assert!(irreps.layout().is_empty());
    }

    #[test]
    fn rejects_invalid_irrep_inputs() {
        for input in ["", "e", "1", "1p", "-1e", "1.0e", "1 e", "1E"] {
            assert!(input.parse::<Irrep>().is_err(), "{input} should fail");
        }
    }

    #[test]
    fn rejects_invalid_irreps_inputs() {
        for input in [
            "",
            " ",
            "+",
            "1x0e +",
            "1x0e ++ 2x1o",
            "0x0e",
            "x0e",
            "1x",
            "1xx0e",
            "1 x0e",
            "1x 0e",
            "2*1o",
            "1x0p",
        ] {
            assert!(input.parse::<Irreps>().is_err(), "{input} should fail");
        }
    }

    #[test]
    fn multiplies_parity() {
        assert_eq!(Parity::Even.multiply(Parity::Even), Parity::Even);
        assert_eq!(Parity::Even.multiply(Parity::Odd), Parity::Odd);
        assert_eq!(Parity::Odd.multiply(Parity::Even), Parity::Odd);
        assert_eq!(Parity::Odd.multiply(Parity::Odd), Parity::Even);
    }
}
