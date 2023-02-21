#
# test the cluster directive
#
library(survival)
fit1 <- aareg(Surv(tstart, tstop, status) ~ treat + age + inherit +
                         steroids, data=cgd)
fit2 <- aareg(Surv(tstart, tstop, status) ~ treat + age + inherit +
                         steroids, data=cgd, cluster=id)
