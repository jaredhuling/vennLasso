#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* .Call entry points (defined in ogLasso.cpp) */
extern SEXP admm_oglasso_dense(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                               SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                               SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"admm_oglasso_dense", (DL_FUNC) &admm_oglasso_dense, 21},
    {NULL, NULL, 0}
};

void R_init_vennLasso(DllInfo *info)
{
    R_registerRoutines(info, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
