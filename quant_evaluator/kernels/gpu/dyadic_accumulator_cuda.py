"""Shared CUDA device helpers for exact sums of finite binary64 values."""

DYADIC_DEVICE_SOURCE = r"""
struct DyadicAccumulator {
  unsigned long long positive[68];
  unsigned long long negative[68];
};

__device__ __forceinline__ void dyadic_init(DyadicAccumulator* a) {
  for (int i=0; i<68; ++i) {
    a->positive[i]=0ULL;
    a->negative[i]=0ULL;
  }
}

// Accumulate coefficient copies of v as an integer multiple of 2^-1074.
// The callers cap total weighted additions at INT32_MAX, so every limb's
// pre-carry sum remains below 2^63 and fits the unsigned 64-bit scratch word.
__device__ __forceinline__ void dyadic_add(
    DyadicAccumulator* a, double v, int coefficient) {
  int repeat=coefficient<0 ? -coefficient : coefficient;
  if (repeat==0 || v==0.0) return;
  unsigned long long bits=__double_as_longlong(v);
  unsigned long long mant=bits & 0x000fffffffffffffULL;
  int exponent=(int)((bits>>52)&0x7ffULL);
  if (exponent) mant |= 0x0010000000000000ULL;
  if (mant==0ULL) return;
  int shift=exponent ? exponent-1 : 0;
  int limb=shift>>5, offset=shift&31;
  unsigned long long lo=mant & 0xffffffffULL;
  unsigned long long hi=mant>>32;
  unsigned long long w0=(lo<<offset)&0xffffffffULL;
  unsigned long long carry0=offset ? (lo>>(32-offset)) : 0ULL;
  unsigned long long w1=((hi<<offset)|carry0)&0xffffffffULL;
  unsigned long long carry1=offset ? (hi>>(32-offset)) : 0ULL;
  int is_negative=((bits>>63)!=0) ^ (coefficient<0);
  unsigned long long* words=is_negative ? a->negative : a->positive;
  for (int n=0; n<repeat; ++n) {
    words[limb]+=w0;
    if (limb+1<68) words[limb+1]+=w1;
    if (carry1 && limb+2<68) words[limb+2]+=carry1;
  }
}

__device__ __forceinline__ double dyadic_quotient_rne(
    DyadicAccumulator* a, unsigned long long denominator) {
  unsigned long long* positive=a->positive;
  unsigned long long* negative=a->negative;
  for (int i=0; i<67; ++i) {
    unsigned long long carry=positive[i]>>32;
    positive[i]&=0xffffffffULL;
    positive[i+1]+=carry;
    carry=negative[i]>>32;
    negative[i]&=0xffffffffULL;
    negative[i+1]+=carry;
  }
  positive[67]&=0xffffffffULL;
  negative[67]&=0xffffffffULL;
  int comparison=0;
  for (int i=67; i>=0; --i) {
    if (positive[i]>negative[i]) { comparison=1; break; }
    if (positive[i]<negative[i]) { comparison=-1; break; }
  }
  if (comparison==0) return 0.0;

  unsigned long long magnitude[68];
  unsigned long long borrow=0ULL;
  for (int i=0; i<68; ++i) {
    unsigned long long x=(comparison>0) ? positive[i] : negative[i];
    unsigned long long y=(comparison>0) ? negative[i] : positive[i];
    unsigned long long sub=y+borrow;
    unsigned long long next=(x<sub);
    magnitude[i]=(x-sub)&0xffffffffULL;
    borrow=next;
  }

  unsigned long long remainder=0ULL;
  for (int i=67; i>=0; --i) {
    unsigned long long current=(remainder<<32)|magnitude[i];
    magnitude[i]=current/denominator;
    remainder=current%denominator;
  }
  int highest=-1;
  for (int i=67; i>=0 && highest<0; --i) {
    if (magnitude[i]) {
      unsigned long long word=magnitude[i];
      int bit=0;
      while (word>>1) { word>>=1; ++bit; }
      highest=i*32+bit;
    }
  }

  unsigned long long outbits=0ULL;
  if (highest<52) {
    unsigned long long significand=0ULL;
    for (int bit=0; bit<52; ++bit)
      if ((magnitude[bit>>5]>>(bit&31))&1ULL) significand|=1ULL<<bit;
    unsigned long long twice=remainder*2ULL;
    if (twice>denominator || (twice==denominator && (significand&1ULL))) ++significand;
    outbits=(significand>=(1ULL<<52)) ? (1ULL<<52) : significand;
  } else {
    int drop=highest-52;
    unsigned long long significand=0ULL;
    for (int bit=0; bit<53; ++bit) {
      int source=drop+bit;
      if ((magnitude[source>>5]>>(source&31))&1ULL) significand|=1ULL<<bit;
    }
    int guard=0, sticky=(remainder!=0ULL);
    if (drop==0) {
      unsigned long long twice=remainder*2ULL;
      if (twice>denominator || (twice==denominator && (significand&1ULL))) ++significand;
    } else {
      int guard_bit=drop-1;
      guard=(int)((magnitude[guard_bit>>5]>>(guard_bit&31))&1ULL);
      for (int bit=0; bit<guard_bit; ++bit) {
        if ((magnitude[bit>>5]>>(bit&31))&1ULL) { sticky=1; break; }
      }
      if (guard && (sticky || (significand&1ULL))) ++significand;
    }
    if (significand==(1ULL<<53)) { significand>>=1; ++highest; }
    int encoded_exponent=highest-51;
    if (encoded_exponent>=2047) outbits=0x7ff0000000000000ULL;
    else outbits=((unsigned long long)encoded_exponent<<52)
                 |(significand&0x000fffffffffffffULL);
  }
  if (comparison<0) outbits|=0x8000000000000000ULL;
  return __longlong_as_double((long long)outbits);
}
"""

