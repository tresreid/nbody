// @(#)root/smatrix:$Id$
// Author: T. Glebe, L. Moneta, J. Palacios    2005

#ifndef ROOT_Math_SMatrix
#define ROOT_Math_SMatrix

#ifndef __CUDACC__
#define __host__
#define __device__
#endif // __CUDACC


/*********************************************************************************
//
// source:
//
// type:      source code
//
// created:   20. Mar 2001
//
// author:    Thorsten Glebe
//            HERA-B Collaboration
//            Max-Planck-Institut fuer Kernphysik
//            Saupfercheckweg 1
//            69117 Heidelberg
//            Germany
//            E-mail: T.Glebe@mpi-hd.mpg.de
//
// Description: A fixed size two dimensional Matrix class
//
// changes:
// 20 Mar 2001 (TG) creation
// 21 Mar 2001 (TG) added operators +=, -=, *=, /=
// 26 Mar 2001 (TG) place_in_row(), place_in_col() added
// 02 Apr 2001 (TG) non-const Array() added
// 03 Apr 2001 (TG) invert() added
// 07 Apr 2001 (TG) CTOR from SVertex (dyadic product) added
// 09 Apr 2001 (TG) CTOR from array added
// 11 Apr 2001 (TG) rows(), cols(), size() replaced by rows, cols, size
// 25 Mai 2001 (TG) row(), col() added
// 04 Sep 2001 (TG) moved inlined functions to .icc file
// 11 Jan 2002 (TG) added operator==(), operator!=()
// 14 Jan 2002 (TG) added more operator==(), operator!=(), operator>(), operator<()
//
***************************************************************************/
// for platform specific configurations

#ifndef ROOT_Math_MnConfig
#include "Math/MConfig.h"
#endif

#include <iosfwd>




//doxygen tag

/**
   @defgroup SMatrixGroup SMatrix

   \ref SMatrix  for high performance vector and matrix computations.
   Classes representing Matrices and Vectors of arbitrary type and dimension and related functions.
   For a detailed description and usage examples see:
   <ul>
    <li>\ref SMatrix home page
    <li>\ref SVectorDoc
    <li>\ref SMatrixDoc
    <li>\ref MatVecFunctions
   </ul>

*/

/**
   @defgroup SMatrixSVector Matrix and Vector classes

   @ingroup SMatrixGroup

   Classes representing Matrices and Vectors of arbitrary type and dimension.
   For a detailed description and usage examples see:
   <ul>
    <li>\ref SVectorDoc
    <li>\ref SMatrixDoc
    <li>\ref MatVecFunctions
   </ul>

*/


#ifndef ROOT_Math_Expression
#include "Math/Expression.h"
#endif
#ifndef ROOT_Math_MatrixRepresentationsStatic
#include "Math/MatrixRepresentationsStatic.h"
#endif


namespace ROOT {

namespace Math {


template <class T, unsigned int D> class SVector;

struct SMatrixIdentity { };
struct SMatrixNoInit { };

//__________________________________________________________________________
/**
    SMatrix: a generic fixed size D1 x D2 Matrix class.
    The class is template on the scalar type, on the matrix sizes:
    D1 = number of rows and D2 = number of columns
    amd on the representation storage type.
    By default the representation is MatRepStd<T,D1,D2> (standard D1xD2 of type T),
    but it can be of type MatRepSym<T,D> for symmetric matrices DxD, where the storage is only
    D*(D+1)/2.

    See \ref SMatrixDoc.

    Original author is Thorsten Glebe
    HERA-B Collaboration, MPI Heidelberg (Germany)

    @ingroup SMatrixSVector

    @authors T. Glebe, L. Moneta and J. Palacios
*/
//==============================================================================
// SMatrix: column-wise storage
//==============================================================================
template <class T,
          unsigned int D1,
          unsigned int D2 = D1,
          class R=MatRepStd<T, D1, D2> >
    __device__ __host__
class SMatrix {
public:
   /** @name --- Typedefs --- */

   /** contained scalar type */
   typedef T  value_type;

   /** storage representation type */
   typedef R  rep_type;

   /** STL iterator interface. */
   typedef T*  iterator;

   /** STL const_iterator interface. */
   typedef const T*  const_iterator;



   /** @name --- Constructors and Assignment --- */

   /**
      Default constructor:
   */
     __device__ __host__
   SMatrix();
   ///
  /**
       construct from without initialization
   */
     __device__ __host__
   inline SMatrix( SMatrixNoInit ){}

   /**
       construct from an identity matrix
   */
     __device__ __host__
   SMatrix( SMatrixIdentity );
   /**
       copy constructor (from a matrix of the same representation
   */
     __device__ __host__
   SMatrix(const SMatrix<T,D1,D2,R>& rhs);
   /**
      construct from a matrix with different representation.
      Works only from symmetric to general and not viceversa.
   */
   template <class R2>
     __device__ __host__
   SMatrix(const SMatrix<T,D1,D2,R2>& rhs);

   /**
      Construct from an expression.
      In case of symmetric matrices does not work if expression is of type general
      matrices. In case one needs to force the assignment from general to symmetric, one can use the
      ROOT::Math::AssignSym::Evaluate function.
   */
   //  __device__ __host__
   template <class A, class R2>
     __device__ __host__
   SMatrix(const Expr<A,T,D1,D2,R2>& rhs);


   /**
      Constructor with STL iterator interface. The data will be copied into the matrix
      \param begin start iterator position
      \param end end iterator position
      \param triang if true only the triangular lower/upper part of the matrix is filled from the iterators
      \param lower if true the lower triangular part is filled

      Size of the matrix must match size of the iterators, if triang is false, otherwise the size of the
      triangular block. In the case of symmetric matrices triang is considered always to be true
      (what-ever the user specifies) and the size of the iterators must be equal to the size of the
      triangular block, which is the number of independent elements of a symmetric matrix:  N*(N+1)/2

   */
   //  __device__ __host__
   template<class InputIterator>
     __device__ __host__
   SMatrix(InputIterator begin, InputIterator end, bool triang = false, bool lower = true);

   /**
      Constructor with STL iterator interface. The data will be copied into the matrix
      \param begin  start iterator position
      \param size   iterator size
      \param triang if true only the triangular lower/upper part of the matrix is filled from the iterators
      \param lower if true the lower triangular part is filled

      Size of the iterators must not be larger than the size of the matrix representation.
      In the case of symmetric matrices the size is N*(N+1)/2.

   */
   //  __device__ __host__
   template<class InputIterator>
     __device__ __host__
   SMatrix(InputIterator begin, unsigned int size, bool triang = false, bool lower = true);

   /**
      constructor of a symmetrix a matrix from a SVector containing the lower (upper)
      triangular part.
   */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
     __device__ __host__
   SMatrix(const SVector<T, D1*(D2+1)/2> & v, bool lower = true );
#else
   template<unsigned int N>
     __device__ __host__
   SMatrix(const SVector<T,N> & v, bool lower = true );
#endif


   /**
       Construct from a scalar value (only for size 1 matrices)
   */
     __device__ __host__
   explicit SMatrix(const T& rhs);

   /**
       Assign from another compatible matrix.
       Possible Symmetirc to general but NOT vice-versa
   */
   template <class M>
     __device__ __host__
   SMatrix<T,D1,D2,R>& operator=(const M& rhs);

   /**
       Assign from a matrix expression
   */
   template <class A, class R2>
     __device__ __host__
   SMatrix<T,D1,D2,R>& operator=(const Expr<A,T,D1,D2,R2>& rhs);

   /**
      Assign from an identity matrix
   */
     __device__ __host__
   SMatrix<T,D1,D2,R> & operator=(SMatrixIdentity );

   /**
       Assign from a scalar value (only for size 1 matrices)
   */
     __device__ __host__
   SMatrix<T,D1,D2,R>& operator=(const T& rhs);

   /** @name --- Matrix dimension --- */

   /**
      Enumeration defining the matrix dimension,
      number of rows, columns and size = rows*columns)
   */
   enum {
      /// return no. of matrix rows
      kRows = D1,
      /// return no. of matrix columns
      kCols = D2,
      /// return no of elements: rows*columns
      kSize = D1*D2
   };

   /** @name --- Access functions --- */

   /** access the parse tree with the index starting from zero and
       following the C convention for the order in accessing
       the matrix elements.
       Same convention for general and symmetric matrices.
   */
     __device__ __host__
   T apply(unsigned int i) const;

   /// return read-only pointer to internal array
     __device__ __host__
   const T* Array() const;
   /// return pointer to internal array
     __device__ __host__
   T* Array();

   /** @name --- STL-like interface ---
       The iterators access the matrix element in the order how they are
       stored in memory. The C (row-major) convention is used, and in the
       case of symmetric matrices the iterator spans only the lower diagonal
       block. For example for a symmetric 3x3 matrices the order of the 6
       elements \f${a_0,...a_5}\f$ is:
       \f[
       M = \left( \begin{array}{ccc}
       a_0 & a_1 & a_3  \\
       a_1 & a_2  & a_4  \\
       a_3 & a_4 & a_5   \end{array} \right)
       \f]
   */

   /** STL iterator interface. */
     __device__ __host__
   iterator begin();

   /** STL iterator interface. */
     __device__ __host__
   iterator end();

   /** STL const_iterator interface. */
     __device__ __host__
   const_iterator begin() const;

   /** STL const_iterator interface. */
     __device__ __host__
   const_iterator end() const;

   /**
      Set matrix elements with STL iterator interface. The data will be copied into the matrix
      \param begin start iterator position
      \param end end iterator position
      \param triang if true only the triangular lower/upper part of the matrix is filled from the iterators
      \param lower if true the lower triangular part is filled

      Size of the matrix must match size of the iterators, if triang is false, otherwise the size of the
      triangular block. In the case of symmetric matrices triang is considered always to be true
      (what-ever the user specifies) and the size of the iterators must be equal to the size of the
      triangular block, which is the number of independent elements of a symmetric matrix:  N*(N+1)/2

   */
   template<class InputIterator>
   __device__ __host__
   void SetElements(InputIterator begin, InputIterator end, bool triang = false, bool lower = true);

   /**
      Constructor with STL iterator interface. The data will be copied into the matrix
      \param begin  start iterator position
      \param size   iterator size
      \param triang if true only the triangular lower/upper part of the matrix is filled from the iterators
      \param lower if true the lower triangular part is filled

      Size of the iterators must not be larger than the size of the matrix representation.
      In the case of symmetric matrices the size is N*(N+1)/2.

   */
   template<class InputIterator>
  __device__ __host__
   void SetElements(InputIterator begin, unsigned int size, bool triang = false, bool lower = true);


   /** @name --- Operators --- */
   /// element wise comparison
     __device__ __host__
   bool operator==(const T& rhs) const;
   /// element wise comparison
     __device__ __host__
   bool operator!=(const T& rhs) const;
   /// element wise comparison
   template <class R2>
     __device__ __host__
   bool operator==(const SMatrix<T,D1,D2,R2>& rhs) const;
   /// element wise comparison
     __device__ __host__
   bool operator!=(const SMatrix<T,D1,D2,R>& rhs) const;
   /// element wise comparison
   template <class A, class R2>
     __device__ __host__
   bool operator==(const Expr<A,T,D1,D2,R2>& rhs) const;
   /// element wise comparison
   template <class A, class R2>
     __device__ __host__
   bool operator!=(const Expr<A,T,D1,D2,R2>& rhs) const;

   /// element wise comparison
     __device__ __host__
   bool operator>(const T& rhs) const;
   /// element wise comparison
     __device__ __host__
   bool operator<(const T& rhs) const;
   /// element wise comparison
   template <class R2>
     __device__ __host__
   bool operator>(const SMatrix<T,D1,D2,R2>& rhs) const;
   /// element wise comparison
   template <class R2>
     __device__ __host__
   bool operator<(const SMatrix<T,D1,D2,R2>& rhs) const;
   /// element wise comparison
   template <class A, class R2>
     __device__ __host__
   bool operator>(const Expr<A,T,D1,D2,R2>& rhs) const;
   /// element wise comparison
   template <class A, class R2>
     __device__ __host__
   bool operator<(const Expr<A,T,D1,D2,R2>& rhs) const;

   /**
      read only access to matrix element, with indices starting from 0
   */
     __device__ __host__
   const T& operator()(unsigned int i, unsigned int j) const;
    /**
      read/write access to matrix element with indices starting from 0
   */
     __device__ __host__
   T& operator()(unsigned int i, unsigned int j);

   /**
      read only access to matrix element, with indices starting from 0.
      Function will check index values and it will assert if they are wrong
   */
     __device__ __host__
   const T& At(unsigned int i, unsigned int j) const;
   /**
      read/write access to matrix element with indices starting from 0.
      Function will check index values and it will assert if they are wrong
   */
     __device__ __host__
   T& At(unsigned int i, unsigned int j);


  // helper class for implementing the m[i][j] operator

   class SMatrixRow {
   public:
     __device__ __host__
      SMatrixRow ( SMatrix<T,D1,D2,R> & rhs, unsigned int i ) :
         fMat(&rhs), fRow(i)
      {}
     __device__ __host__
      T & operator[](int j) { return (*fMat)(fRow,j); }
   private:
      SMatrix<T,D1,D2,R> *  fMat;
      unsigned int fRow;
   };

   class SMatrixRow_const {
   public:
     __device__ __host__
      SMatrixRow_const ( const SMatrix<T,D1,D2,R> & rhs, unsigned int i ) :
         fMat(&rhs), fRow(i)
      {}

     __device__ __host__
      const T & operator[](int j) const { return (*fMat)(fRow, j); }

   private:
      const SMatrix<T,D1,D2,R> *  fMat;
      unsigned int fRow;
   };

  /**
      read only access to matrix element, with indices starting from 0 : m[i][j]
   */
     __device__ __host__
   SMatrixRow_const  operator[](unsigned int i) const { return SMatrixRow_const(*this, i); }
   /**
      read/write access to matrix element with indices starting from 0 : m[i][j]
   */
     __device__ __host__
   SMatrixRow operator[](unsigned int i) { return SMatrixRow(*this, i); }


   /**
      addition with a scalar
   */
     __device__ __host__
   SMatrix<T,D1,D2,R>&operator+=(const T& rhs);

   /**
      addition with another matrix of any compatible representation
   */
   template <class R2>
     __device__ __host__
   SMatrix<T,D1,D2,R>&operator+=(const SMatrix<T,D1,D2,R2>& rhs);

   /**
      addition with a compatible matrix expression
   */
   template <class A, class R2>
     __device__ __host__
   SMatrix<T,D1,D2,R>& operator+=(const Expr<A,T,D1,D2,R2>& rhs);

   /**
      subtraction with a scalar
   */
     __device__ __host__
   SMatrix<T,D1,D2,R>& operator-=(const T& rhs);

   /**
      subtraction with another matrix of any compatible representation
   */
   template <class R2>
     __device__ __host__
   SMatrix<T,D1,D2,R>&operator-=(const SMatrix<T,D1,D2,R2>& rhs);

   /**
      subtraction with a compatible matrix expression
   */
   template <class A, class R2>
     __device__ __host__
   SMatrix<T,D1,D2,R>& operator-=(const Expr<A,T,D1,D2,R2>& rhs);

   /**
      multiplication with a scalar
    */
     __device__ __host__
   SMatrix<T,D1,D2,R>& operator*=(const T& rhs);

#ifndef __CINT__


   /**
      multiplication with another compatible matrix (it is a real matrix multiplication)
      Note that this operation does not avid to create a temporary to store intermidiate result
   */
   template <class R2>
     __device__ __host__
   SMatrix<T,D1,D2,R>& operator*=(const SMatrix<T,D1,D2,R2>& rhs);

   /**
      multiplication with a compatible matrix expression (it is a real matrix multiplication)
   */
   template <class A, class R2>
     __device__ __host__
   SMatrix<T,D1,D2,R>& operator*=(const Expr<A,T,D1,D2,R2>& rhs);

#endif

   /**
      division with a scalar
   */
     __device__ __host__
   SMatrix<T,D1,D2,R>& operator/=(const T& rhs);



   /** @name --- Linear Algebra Functions --- */

   /**
      Invert a square Matrix ( this method changes the current matrix).
      Return true if inversion is successfull.
      The method used for general square matrices is the LU factorization taken from Dinv routine
      from the CERNLIB (written in C++ from CLHEP authors)
      In case of symmetric matrices Bunch-Kaufman diagonal pivoting method is used
      (The implementation is the one written by the CLHEP authors)
   */
     __device__ __host__
   bool Invert();

   /**
      Invert a square Matrix and  returns a new matrix. In case the inversion fails
      the current matrix is returned.
      \param ifail . ifail will be set to 0 when inversion is successfull.
      See ROOT::Math::SMatrix::Invert for the inversion algorithm
   */
     __device__ __host__
   SMatrix<T,D1,D2,R> Inverse(int & ifail ) const;

   /**
      Fast Invertion of a square Matrix ( this method changes the current matrix).
      Return true if inversion is successfull.
      The method used is based on direct inversion using the Cramer rule for
      matrices upto 5x5. Afterwards the same default algorithm of Invert() is used.
      Note that this method is faster but can suffer from much larger numerical accuracy
      when the condition of the matrix is large
   */
     __device__ __host__
   bool InvertFast();

   /**
      Invert a square Matrix and  returns a new matrix. In case the inversion fails
      the current matrix is returned.
      \param ifail . ifail will be set to 0 when inversion is successfull.
      See ROOT::Math::SMatrix::InvertFast for the inversion algorithm
   */
     __device__ __host__
   SMatrix<T,D1,D2,R> InverseFast(int & ifail ) const;

   /**
      Invertion of a symmetric positive defined Matrix using Choleski decomposition.
      ( this method changes the current matrix).
      Return true if inversion is successfull.
      The method used is based on Choleski decomposition
      A compile error is given if the matrix is not of type symmetric and a run-time failure if the
      matrix is not positive defined.
      For solving  a linear system, it is possible to use also the function
      ROOT::Math::SolveChol(matrix, vector) which will be faster than performing the inversion
   */
     __device__ __host__
   bool InvertChol();

   /**
      Invert of a symmetric positive defined Matrix using Choleski decomposition.
      A compile error is given if the matrix is not of type symmetric and a run-time failure if the
      matrix is not positive defined.
      In case the inversion fails the current matrix is returned.
      \param ifail . ifail will be set to 0 when inversion is successfull.
      See ROOT::Math::SMatrix::InvertChol for the inversion algorithm
   */
     __device__ __host__
   SMatrix<T,D1,D2,R> InverseChol(int & ifail ) const;

   /**
      determinant of square Matrix via Dfact.
      Return true when the calculation is successfull.
      \param det will contain the calculated determinant value
      \b Note: this will destroy the contents of the Matrix!
   */
     __device__ __host__
   bool Det(T& det);

   /**
      determinant of square Matrix via Dfact.
      Return true when the calculation is successfull.
      \param det will contain the calculated determinant value
      \b Note: this will preserve the content of the Matrix!
   */
     __device__ __host__
   bool Det2(T& det) const;


   /** @name --- Matrix Slice Functions --- */

   /// place a vector in a Matrix row
   template <unsigned int D>
     __device__ __host__
   SMatrix<T,D1,D2,R>& Place_in_row(const SVector<T,D>& rhs,
                                    unsigned int row,
                                    unsigned int col);
   /// place a vector expression in a Matrix row
   template <class A, unsigned int D>
     __device__ __host__
   SMatrix<T,D1,D2,R>& Place_in_row(const VecExpr<A,T,D>& rhs,
                                    unsigned int row,
                                    unsigned int col);
   /// place a vector in a Matrix column
   template <unsigned int D>
     __device__ __host__
   SMatrix<T,D1,D2,R>& Place_in_col(const SVector<T,D>& rhs,
                                    unsigned int row,
                                    unsigned int col);
   /// place a vector expression in a Matrix column
   template <class A, unsigned int D>
     __device__ __host__
   SMatrix<T,D1,D2,R>& Place_in_col(const VecExpr<A,T,D>& rhs,
                                    unsigned int row,
                                    unsigned int col);
   /// place a matrix in this matrix
   template <unsigned int D3, unsigned int D4, class R2>
     __device__ __host__
   SMatrix<T,D1,D2,R>& Place_at(const SMatrix<T,D3,D4,R2>& rhs,
                                unsigned int row,
                                unsigned int col);
   /// place a matrix expression in this matrix
   template <class A, unsigned int D3, unsigned int D4, class R2>
     __device__ __host__
   SMatrix<T,D1,D2,R>& Place_at(const Expr<A,T,D3,D4,R2>& rhs,
                                unsigned int row,
                                unsigned int col);

   /**
      return a full Matrix row as a vector (copy the content in a new vector)
   */
     __device__ __host__
   SVector<T,D2> Row(unsigned int therow) const;

   /**
      return a full Matrix column as a vector (copy the content in a new vector)
   */
     __device__ __host__
   SVector<T,D1> Col(unsigned int thecol) const;

   /**
      return a slice of therow as a vector starting at the colum value col0 until col0+N,
      where N is the size of the vector (SubVector::kSize )
      Condition  col0+N <= D2
   */
   template <class SubVector>
     __device__ __host__
   SubVector SubRow(unsigned int therow, unsigned int col0 = 0 ) const;

   /**
      return a slice of the column as a vector starting at the row value row0 until row0+Dsub.
      where N is the size of the vector (SubVector::kSize )
      Condition  row0+N <= D1
   */
   template <class SubVector>
     __device__ __host__
   SubVector SubCol(unsigned int thecol, unsigned int row0 = 0) const;

   /**
      return a submatrix with the upper left corner at the values (row0, col0) and with sizes N1, N2
      where N1 and N2 are the dimension of the sub-matrix (SubMatrix::kRows and SubMatrix::kCols )
      Condition  row0+N1 <= D1 && col0+N2 <=D2
   */
   template <class SubMatrix >
     __device__ __host__
   SubMatrix Sub(unsigned int row0, unsigned int col0) const;

   /**
      return diagonal elements of a matrix as a Vector.
      It works only for squared matrices D1 == D2, otherwise it will produce a compile error
   */
     __device__ __host__
   SVector<T,D1> Diagonal() const;

   /**
      Set the diagonal elements from a Vector
      Require that vector implements ::kSize since a check (statically) is done on
      diagonal size == vector size
   */
   template <class Vector>
     __device__ __host__
   void SetDiagonal(const Vector & v);

   /**
      return the trace of a matrix
      Sum of the diagonal elements
   */
     __device__ __host__
   T Trace() const;


   /**
      return the upper Triangular block of the matrices (including the diagonal) as
      a vector of sizes N = D1 * (D1 + 1)/2.
      It works only for square matrices with D1==D2, otherwise it will produce a compile error
   */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
     __device__ __host__
   SVector<T, D1 * (D2 +1)/2> UpperBlock() const;
#else
   template<class SubVector>
     __device__ __host__
   SubVector UpperBlock() const;
#endif

   /**
      return the lower Triangular block of the matrices (including the diagonal) as
      a vector of sizes N = D1 * (D1 + 1)/2.
      It works only for square matrices with D1==D2, otherwise it will produce a compile error
   */
#ifndef UNSUPPORTED_TEMPLATE_EXPRESSION
     __device__ __host__
   SVector<T, D1 * (D2 +1)/2> LowerBlock() const;
#else
     __device__ __host__
   template<class SubVector>
   SubVector LowerBlock() const;
#endif


   /** @name --- Other Functions --- */

   /**
       Function to check if a matrix is sharing same memory location of the passed pointer
       This function is used by the expression templates to avoid the alias problem during
       expression evaluation. When  the matrix is in use, for example in operations
       like A = B * A, a temporary object storing the intermediate result is automatically
       created when evaluating the expression.

   */
     __device__ __host__
   bool IsInUse(const T* p) const;

   // submatrices

   /// Print: used by operator<<()
     __device__ __host__
   std::ostream& Print(std::ostream& os) const;




public:

   /** @name --- Data Member --- */

   /**
      Matrix Storage Object containing matrix data
   */
   R fRep;

}; // end of class SMatrix




//==============================================================================
// operator<<
//==============================================================================
template <class T, unsigned int D1, unsigned int D2, class R>
     __device__ __host__
inline std::ostream& operator<<(std::ostream& os, const ROOT::Math::SMatrix<T,D1,D2,R>& rhs) {
  return rhs.Print(os);
}


  }  // namespace Math

}  // namespace ROOT






#ifndef __CINT__

#ifndef ROOT_Math_SMatrix_icc
#include "./SMatrix.icc"
#endif

#ifndef ROOT_Math_MatrixFunctions
#include "Math/MatrixFunctions.h"
#endif

#endif //__CINT__

#endif  /* ROOT_Math_SMatrix  */
