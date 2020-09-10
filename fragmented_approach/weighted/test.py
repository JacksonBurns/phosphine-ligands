import matplotlib.pyplot as plt
ez_ratio = [
    1.257691892,0.143694327,2.979755179,1.546341135,1.96865129,1.289964041,1.344086022,1.322926522,1.365829346,2.139425287,0.712878788,0.653353659,
    1.695454545,0.510582011,0.645833333,3.847132285,0.365384615,1.425646552,0.132211943,1.310784314,1,1.036119711,1.326143791,3.348148148,8.646451613,
    2.058915771,8.042063492,6.145935961,3.408510638,2.346433919,4.666666667,2.990291262,5.446864381,6.368269231,4.308321479,1.601515152,4.045152629,
    5.033333333,3.125636672,3.80466996,0.652380952,3.956521739,4.406296296,1.229166667,3.008651026,2.597826087,1.719907407,1.346945796,4.183395872,2.023701807,
    2.594594595
    ]
new_reatio = [1.526879485,0.444732371,5.030074725,1.608059136,1.685125644,1.543219337,2.437720143,1.500097106,1.253291838,1.393147294,0.779434044,4.970690933,3.656148516,1.432638105,4.392882149,0.322052668,4.342422536,2.662450109,2.025653956,1.819881262,4.942379568,1.541499959,1.554145527,2.696659327,6.87079137,1.596468609,2.361179505,3.205079971,3.114671716,3.174870162,2.881574165,3.273573207,2.975114512,2.704627303,2.613073625,0.68830987,3.300680904,2.992386038,1.280113662,2.92600431,1.594979045,2.447493015,3.412007504,2.533187017,2.12504873,1.72051235,1.496295,1.493525871,2.469264493,1.446522925,2.072528865]
print(max(ez_ratio))
print(max(new_reatio))
plt.figure()
plt.hist([ez_ratio, new_reatio],rwidth=0.9,bins=[i/2 for i in range(0,19)],align='left',label=['E:Z','2:1'])
plt.legend(loc='best')
plt.title('E:Z vs. 2:1 Distribution of Values')
plt.ylabel('Frequency')
plt.xlabel('Selectivity')
plt.xticks([i/2 for i in range(0,19)],[str(round(i/2,2)) for i in range(0,19)])
plt.show()