#pragma once 
template<class T>
class max_OP{
  public:
  static inline T OP(T& v1, T& v2){
      if ( v2 < v1 )
        return v1;
      return v2;
  }

  template<class P>
  static bool validate(T val, P correct){
    if ( val == static_cast<T>(correct-1) )
      return true;
    return false; 
  }

  static T init(){
    return std::numeric_limits<T>::min(); 
  }

  static T init(long i){
    return static_cast<T>(i);
  }
};

template<class T>
class min_OP{
  public:
  static inline T OP(T& v1, T& v2){
      if ( v2 > v1 )
        return v1;
      return v2;
  }

  template<class P>
  static bool validate(T val, P val1){
    if ( val == static_cast<T>(0) )
      return true;
    return false; 
  }

  static T init(){
    return std::numeric_limits<T>::max(); 
  }

  static T init(long i){
    return static_cast<T>(i);
  }
};

template<class T>
class add_OP{
  public:
  static inline T OP(T& v1, T& v2){
      return v1 + v2;
  }

  template<class P>
  static bool validate(T val, P val1){
    T tmp = static_cast<T>(val1);
    T tmp1 = tmp/2.0;
    T  S = tmp/2.0 * (tmp-1);
    if ( val == S )
      return true;
    return false; 
  }

  static T init(){
    return 0; 
  }

  static T init(long i){
    return static_cast<T>(i);
  }
};

template<class T>
class mult_OP{
  public:
  static inline T OP(T& v1, T& v2){
      return v1 * v2;
  }

  template<class P>
  static bool validate(T val, P val1){
    if ( val == 1.0 )
      return true;
    return false; 
  }

  static T init(){
    return 1; 
  }

  static T init(long i){
    return 1;
  }
};

