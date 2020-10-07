from datetime import date
from newENV import BS
today = date.today()
print("Today's date:"+ str(today))
#--------------------------------------------------------------------------
env = BS(nBS=4,nUE=4,nMaxLink=2,nFile=5,nMaxCache=2,loadENV = True)
titleNmae = 'Energy Efficiency \n nBS='+str(env.B)+ \
                                    ',nUE='+str(env.U)+\
                                    ',nMaxLink='+str(env.maxLink)+\
                                    ',nFile='+str(env.F)+\
                                    ',nMaxCache='+str(env.N)
print(titleNmae)
#--------------------------------------------------------------------------
