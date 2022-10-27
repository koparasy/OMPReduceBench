#pragma once

#ifndef __OPEN_MP__
#define __HD__ __host__ __device__
#else
#define __HD__
#endif


template<class T>
class MAX{
  public:
#ifdef __RAJA__
  using reduction_type = RAJA::ReduceMax<RAJA::cuda_reduce, T>;
#endif

#ifndef __RAJA__
  __HD__ static inline T OP(T& v1, T& v2){
      if ( v2 < v1 )
        return v1;
      return v2;
  }

  __HD__ static inline T OP(const T& v1, const T& v2){
      if ( v2 < v1 )
        return v1;
      return v2;
  }
#else
  template <class R>
  __HD__ static inline void OP(R& v1, T& v2){
    v1.max(v2);
  }

  template <class R>
  __HD__ static inline void OP(const R& v1, const T& v2){
    v1.max(v2);
  }
#endif

  template<class P>
  __HD__ static T correct(P val1){
    return static_cast<T>(val1/2 -1);
  }

  template<class P>
  __HD__ static bool validate(T val, P correct){
    correct = correct/2 - 1;
    if ( val == static_cast<T>(correct) )
      return true;
    return false; 
  }

  __HD__ static T init(){
    return std::numeric_limits<T>::min(); 
  }

  __HD__ static T init(long i, long elements){
    return static_cast<T>(i - (elements/2));
  }

  __HD__ static const char *info(){
    return "max";
  }
};

template<class T>
class MIN{
  public:
#ifdef __RAJA__
  using reduction_type = RAJA::ReduceMin<RAJA::cuda_reduce, T>;
#endif

#ifndef __RAJA__
  __HD__ static inline T OP(T& v1, T& v2){
      if ( v2 > v1 )
        return v1;
      return v2;
  }

  __HD__ static inline T OP(const T& v1, const T& v2){
      if ( v2 > v1 )
        return v1;
      return v2;
  }
#else
  template <class R>
  __HD__ static inline void OP(R& v1, T& v2){
    v1.min(v2);
  }

  template <class R>
  __HD__ static inline void OP(const R& v1, const T& v2){
    v1.min(v2);
  }
#endif

  template<class M>
  __HD__ static inline M OP(const M& v1, const T& v2){
      if ( v2 < v1 )
        return v1;
      return v2;
  }


  template<class P>
  __HD__ static T correct(P val1){
    return static_cast<T>(-val1/2);
  }

  template<class P>
  __HD__ static bool validate(T val, P val1){
    val1 = -val1/2;
    if ( val == static_cast<T>(val1) )
      return true;
    return false;
  }

  __HD__ static T init(){
    return std::numeric_limits<T>::max();
  }

  __HD__ static T init(long i, long elements){
    return static_cast<T>(i - (elements/2));
  }

  __HD__ static const char *info(){
    return "min";
  }
};

template<class T>
class ADD{
  public:
#ifdef __RAJA__
  using reduction_type = RAJA::ReduceSum<RAJA::cuda_reduce, T>;
#endif

#ifndef __RAJA__
  __HD__ static inline T OP(T& v1, T& v2){
      return v1 + v2;
  }

  __HD__ static inline T OP(const T& v1, const T& v2){
      return v1 + v2;
  }
#else
  template <class R>
  __HD__ static inline void OP(const R& v1, const T& v2){
    v1 +=v2;
  }

  template <class R>
  __HD__ static inline void OP(R& v1, T& v2){
    v1 +=v2;
  }
#endif

  template<class P>
  __HD__ static bool validate(T val, P val1){
    P tmp = -val1/2;
    if ( val == ((val1-1)%2) * tmp )
      return true;
    return false;
  }

  template<class P>
  __HD__ static T correct(P val1){
    P tmp = -val1/2;
    return ((val1-1)%2) * tmp ;
  }

  __HD__ static T init(){
    return 0;
  }

  __HD__ static T init(long i, long elements){
    return static_cast<T>(i  - (elements/2));
  }

  __HD__  static const char *info(){
    return "add";
  }
};

template<class T>
class MUL{
  public:
#ifdef __RAJA__
  using reduction_type = RAJA::ReduceSum<RAJA::cuda_reduce, T>;
#endif

#ifndef __RAJA__
  __HD__ static inline T OP(T& v1, T& v2){
      return v1 * v2;
  }

  __HD__ static inline T OP(const T& v1, const T& v2){
      return v1 * v2;
  }
#else
  template <class R>
  __HD__ static inline void OP(const R& v1, const T& v2){
    v1 *=v2;
  }

  template <class R>
  __HD__ static inline void OP(R& v1, T& v2){
    v1 *=v2;
  }
#endif

  template<class P>
  __HD__ static bool validate(T val, P val1){
    if ( val == 1.0 )
      return true;
    return false;
  }

  template<class P>
  __HD__ static T correct(P val1){
    return static_cast<T>(1);
  }

  __HD__ static T init(){
    return 1;
  }

  __HD__ static T init(long i, long elements ){
    return 1;
  }

  __HD__ static const char *info(){
    return "mult";
  }
};

