import numpy as np
import scipy.stats as stats

def ICC(matrix, alpha=0.05, r0=0):
    '''Intraclass correlation
    matrix is matrix of observations. Each row is an object of measurement and
    each column is a judge or measurement.
    '1-1' is implement: The degree of absolute agreement among measurements made on
    randomly selected objects. It estimates the correlation of any two
    measurements.
    ICC is the estimated intraclass correlation. LB and UB are upper
    and lower bounds of the ICC with alpha level of significance. 

    In addition to estimation of ICC, a hypothesis test is performed
    with the null hypothesis that ICC = r0. The F value, degrees of
    freedom and the corresponding p-value of the this test are reported.

    Translated in python from the matlab code of Arash Salarian, 2008

    Reference: McGraw, K. O., Wong, S. P., "Forming Inferences About
    Some Intraclass Correlation Coefficients", Psychological Methods,
    Vol. 1, No. 1, pp. 30-46, 1996'''

    M = np.array(matrix)
    n, k = np.shape(M)
    SStotal = np.var(M.flatten(), ddof=1) * (n * k - 1)
    MSR = np.var(np.mean(M, 1), ddof=1) * k
    MSW = np.sum(np.var(M, 1, ddof=1)) / n
    MSC = np.var(np.mean(M, 0), ddof=1) * n
    MSE = (SStotal - MSR * (n - 1) - MSC * (k - 1)) / ((n - 1) * (k - 1))
    # print(n, k)
    # print("SStotal", SStotal)
    # print("MSR", MSR)
    # print("MSW", MSW)
    # print("MSC", MSC)
    # print("MSE", MSE)

    r = (MSR - MSW) / (MSR + (k - 1) * MSW)

    F = (MSR / MSW) * (1 - r0) / (1 + (k - 1) * r0)
    df1 = n - 1
    df2 = n * (k - 1)
    p = 1 - stats.f.cdf(F, df1, df2)

    FL = (MSR / MSW) * (stats.f.isf(1 - alpha / 2, n * (k - 1), n - 1))
    FU = (MSR / MSW) / (stats.f.isf(1 - alpha / 2, n - 1, n * (k - 1)))

    LB = (FL - 1) / (FL + (k - 1))
    UB = (FU - 1) / (FU + (k - 1))
    return (r, LB, UB, F, df1, df2, p)