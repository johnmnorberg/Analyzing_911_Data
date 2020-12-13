# "Analyzing 9-1-1 Call Data using Bayesian Regression"

# This is an analysis of Poisson and negative binomial regression
# models using both Bayesian and frequentist methods. The data is 
# aggregated data from a Whitcom 9-1-1 public records request as 
# well as additional information regarding home football games for 
# Washington State University. Of the five columns in the data set 
# (Count, Date, Hour, Month, Game.Day), all but Date are of interest
# for this analysis. Using Count as the response, we will fit four 
# models using a combination of Hour, Month, and Game.Day. All in 
# all, there will be sixteen models evaluated.

library(rjags)
library(ggplot2)
library(ggpubr)
library(MASS)

# Read in the data
data = read.csv('Data/Project_data.csv', header=TRUE)
data$Hour = as.factor(data$Hour)
data$Month = as.factor(data$Month)
data$Game.Day = as.factor(data$Game.Day)


# Histogram of Count
ggplot(data, aes(x=Count))+geom_histogram(bins=44)+
  ylab('Frequency of Count')+ggtitle('Histogram of Count')


# Plot log(Count) against each predictor
ggarrange(
  ggplot(data, aes(x=Hour, y=log(Count)))+geom_boxplot()+
    xlab('Hour')+ylab('Log(Count)'),
  
  ggplot(data, aes(x=Month, y=log(Count)))+geom_boxplot()+
    xlab('Month')+ylab('Log(Count)'),
  
  ggplot(data, aes(x=Game.Day, y=log(Count)))+geom_boxplot()+
    xlab('Game.Day')+ylab('Log(Count)'))


# It appears there may exist predictive relationships between
# the log of Count and each predictor.


# Create the response matrix
Y = data$Count


# Create the design matrix
X = model.matrix(Count~Hour+Month+Game.Day, data=data)
head(X)


set.seed(7)


# 80/20 training/testing split
train_rows = sample(nrow(data), 0.8*nrow(data))


# Matrix form
train_X = X[train_rows,]
test_X = X[-train_rows,]

train_Y = Y[train_rows]
test_Y = Y[-train_rows]


# Dataframe form
train_data = data[train_rows,]
test_data = data[-train_rows,]


# Poisson models and priors
p_jags = '
model{
  for(i in 1:n){
  
    # LIKELIHOODS
  
    # Model 1 (Count~Hour)
    Y1[i] ~ dpois(lambda1[i])
    log(lambda1[i]) <- inprod(X1[i,], beta1[])
    
    # Model 2 (Count~Hour+Month)
    Y2[i] ~ dpois(lambda2[i])
    log(lambda2[i]) <- inprod(X2[i,], beta2[])
    
    # Model 3 (Count~Hour+Game.Day)
    Y3[i] ~ dpois(lambda3[i])
    log(lambda3[i]) <- inprod(X3[i,], beta3[])
    
    # Model 4 (Count~Hour+Month+Game.Day)
    Y4[i] ~ dpois(lambda4[i])
    log(lambda4[i]) <- inprod(X4[i,], beta4[])
  }
  
  
  
  # PRIORS 
  
  # Priors for Model 1
  for(p1 in 1:12){
    beta1[p1] ~ dnorm(0, 0.001)
  }
    
  # Priors for Model 2
  for(p2 in 1:15){
    beta2[p2] ~ dnorm(0, 0.001)
  }
  
  # Priors for Model 3
  for(p3 in 1:13){
    beta3[p3] ~ dnorm(0, 0.001)
  }
  
  # Priors for Model 4
  for(p4 in 1:16){
    beta4[p4] ~ dnorm(0, 0.001)
  }


  
  # PREDICTIONS USING THE TEST SET
  
	for(i in 1:m){
	
	  # Model 1 Predictions
	  log(lambda1_star[i]) <- inprod(X1_test[i,], beta1[])
		pred1[i] ~ dpois(lambda1_star[i])
		
		# Model 2 Predictions
		log(lambda2_star[i]) <- inprod(X2_test[i,], beta2[])
		pred2[i] ~ dpois(lambda2_star[i])
		
		# Model 3 Predictions
		log(lambda3_star[i]) <- inprod(X3_test[i,], beta3[])
		pred3[i] ~ dpois(lambda3_star[i])
		
		# Model 4 Predictions
		log(lambda4_star[i]) <- inprod(X4_test[i,], beta4[])
		pred4[i] ~ dpois(lambda4_star[i])
	}
}
'


# Data
p_data = list(n=nrow(train_X),
              m=nrow(test_X),
              
              X1=train_X[,1:12],
              X2=train_X[,1:15],
              X3=train_X[,c(1:12, 16)],
              X4=train_X,
              
              X1_test=test_X[,1:12],
              X2_test=test_X[,1:15],
              X3_test=test_X[,c(1:12, 16)],
              X4_test=test_X,
              
              Y1=train_Y,
              Y2=train_Y,
              Y3=train_Y,
              Y4=train_Y)


# Initialize models
p_models = jags.model(file=textConnection(p_jags), data=p_data,
                      inits=list(.RNG.name = 'base::Wichmann-Hill',
                                 .RNG.seed =7))


# Burn in
update(p_models, n.iter=1000)


# Number of iterations
iter = 1e4


# Sample
p_outputs = coda.samples(p_models,
                         variable.names = c('beta1', 'beta2', 
                                            'beta3', 'beta4',
                                            'lambda1', 'lambda2',
                                            'lambda3', 'lambda4',
                                            'pred1', 'pred2',
                                            'pred3', 'pred4'),
                         n.iter = iter)


# Create 4 frequentist models
p1 = glm(Count~Hour, data=train_data, family='poisson')
p2 = glm(Count~Hour+Month, data=train_data, family='poisson')
p3 = glm(Count~Hour+Game.Day, data=train_data, family='poisson')
p4 = glm(Count~Hour+Month+Game.Day, data=train_data, family='poisson')



# Extract the estimate of each Bayesian node
p_est = summary(p_outputs)$statistics[,1]


# Extract the 95% equi-tailed credible set of each node
p_quants = summary(p_outputs)$quantiles[,c(1,5)]


# Create a table of Bayesian coefficients with their credible set
p_beta1 = cbind(Estimate=p_est[1:12], p_quants[1:12,])

p_beta2 = cbind(Estimate=p_est[13:27], p_quants[13:27,])

p_beta3 = cbind(Estimate=p_est[28:40], p_quants[28:40,])

p_beta4 = cbind(p_est[41:56], p_quants[41:56,])


# Rename row names for better interpretation
rownames(p_beta1) = c('(Intercept)', 'Hour13', 'Hour14', 'Hour15', 'Hour16',
                      'Hour17', 'Hour18', 'Hour19', 'Hour20', 'Hour21', 
                      'Hour22', 'Hour23')

rownames(p_beta2) = c('(Intercept)', 'Hour13', 'Hour14', 'Hour15', 'Hour16',
                      'Hour17', 'Hour18', 'Hour19', 'Hour20', 'Hour21', 
                      'Hour22', 'Hour23', 'Month10', 'Month11', 'Month12')

rownames(p_beta3) = c('(Intercept)', 'Hour13', 'Hour14', 'Hour15', 'Hour16', 
                      'Hour17', 'Hour18', 'Hour19', 'Hour20', 'Hour21', 
                      'Hour22', 'Hour23', 'Game.Day')

rownames(p_beta4) = c('(Intercept)', 'Hour13', 'Hour14', 'Hour15', 'Hour16', 
                      'Hour17', 'Hour18', 'Hour19', 'Hour20', 'Hour21', 
                      'Hour22', 'Hour23', 'Month10', 'Month11', 'Month12',
                      'Game.Day')


# Print Bayesian coefficients with their credible sets
p_beta1
p_beta2
p_beta3
p_beta4


# Print GLM coefficients with 95% confidence intervals
cbind(Estimate=coef(p1), confint(p1))
cbind(Estimate=coef(p2), confint(p2))
cbind(Estimate=coef(p3), confint(p3))
cbind(Estimate=coef(p4), confint(p4))


# Poisson deviance residuals
# d_i=sign(Y_i-\lambda_i)\sqrt{2[Y_i\log(Y_i/\lambda_i)
# -(Y_i-\lambda_i)]}
  

# Extract fitted values for Bayesian models
p_fv1 = p_est[57:564]
p_fv2 = p_est[565:1072]
p_fv3 = p_est[1073:1580]
p_fv4 = p_est[1581:2088]


# Calculate deviance residuals for Bayesian models
pois_dev_res = function(fv){
  dr = sign(train_Y-fv)*sqrt(2*(train_Y*log(train_Y/fv)-(train_Y-fv)))
  return(dr)
}

p_dr = data.frame(
  p_dr1 = pois_dev_res(p_fv1),
  p_dr2 = pois_dev_res(p_fv2),
  p_dr3 = pois_dev_res(p_fv3),
  p_dr4 = pois_dev_res(p_fv4),
  row.names = c(1:508))


# Residual analysis
# Residuals vs fitted values
ggarrange(
  ggplot(p_dr, aes(x=p_fv1, y=p_dr1))+geom_point()+
    xlab('Fitted Values')+ylab('Residuals')+
    ggtitle('Bayesian Model 1'),
  
  ggplot(p_dr, aes(x=p_fv2, y=p_dr2))+geom_point()+
    xlab('Fitted Values')+ylab('Residuals')+
    ggtitle('Bayesian Model 2'),
  
  ggplot(p_dr, aes(x=p_fv3, y=p_dr3))+geom_point()+
    xlab('Fitted Values')+ylab('Residuals')+
    ggtitle('Bayesian Model 3'),
  
  ggplot(p_dr, aes(x=p_fv4, y=p_dr4))+geom_point()+
    xlab('Fitted Values')+ylab('Residuals')+
    ggtitle('Bayesian Model 4'))


ggarrange(
  ggplot(p1, aes(p1$fitted.values, p1$residuals))+
    geom_point()+xlab('Fitted Values')+
    ylab('Residuals')+ggtitle('Frequentist Model 1'),
  
  ggplot(p2, aes(p2$fitted.values, p2$residuals))+
    geom_point()+xlab('Fitted Values')+
    ylab('Residuals')+ggtitle('Frequentist Model 2'),
  
  ggplot(p3, aes(p3$fitted.values, p3$residuals))+
    geom_point()+xlab('Fitted Values')+
    ylab('Residuals')+ggtitle('Frequentist Model 3'),
  
  ggplot(p4, aes(p4$fitted.values, p4$residuals))+
    geom_point()+xlab('Fitted Values')+
    ylab('Residuals')+ggtitle('Frequentist Model 4'))


# Histogram of the Bayesian residuals
ggarrange(
  ggplot(p_dr, aes(x=p_dr1))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Bayesian Model 1'),
  
  ggplot(p_dr, aes(x=p_dr2))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Bayesian Model 2'),
  
  ggplot(p_dr, aes(x=p_dr3))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Bayesian Model 3'),
  
  ggplot(p_dr, aes(x=p_dr4))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Bayesian Model 4'))


# Histogram of frequentist residuals
ggarrange(
  ggplot(p1, aes(x=p1$residuals))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Frequentist Model 1'),
  
  ggplot(p2, aes(x=p2$residuals))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Frequentist Model 2'),
  
  ggplot(p3, aes(x=p3$residuals))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Frequentist Model 3'),
  
  ggplot(p4, aes(x=p4$residuals))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Frequentist Model 4'))


# QQ plot of the Bayesian residuals
ggarrange(ggplot(p_dr, aes(sample=p_dr1))+stat_qq()+stat_qq_line()+
            ggtitle('Bayesian Model 1'),
          
          ggplot(p_dr, aes(sample=p_dr2))+stat_qq()+stat_qq_line()+
            ggtitle('Bayesian Model 2'),
          
          ggplot(p_dr, aes(sample=p_dr3))+stat_qq()+stat_qq_line()+
            ggtitle('Bayesian Model 3'),
          
          ggplot(p_dr, aes(sample=p_dr4))+stat_qq()+stat_qq_line()+
            ggtitle('Bayesian Model 4'))


# Histogram of the frequentist residuals
ggarrange(ggplot(p1, aes(sample=p1$residuals))+stat_qq()+stat_qq_line()+
            ggtitle('Frequentist Model 1'),
          
          ggplot(p2, aes(sample=p2$residuals))+stat_qq()+stat_qq_line()+
            ggtitle('Frequentist Model 2'),
          
          ggplot(p3, aes(sample=p3$residuals))+stat_qq()+stat_qq_line()+
            ggtitle('Frequentist Model 3'),
          
          ggplot(p4, aes(sample=p4$residuals))+stat_qq()+stat_qq_line()+
            ggtitle('Frequentist Model 4'))


# Calculate the deviance of each Bayesian model
p_d1 = sum(p_dr$p_dr1^2)
p_d2 = sum(p_dr$p_dr2^2)
p_d3 = sum(p_dr$p_dr3^2)
p_d4 = sum(p_dr$p_dr4^2)


# Summary table (deviance, p-value, dispersion)
p_gof = data.frame(
  Deviance_B = c(p_d1, p_d2, p_d3, p_d4),
  
  GOF_B = c(1-pchisq(p_d1, summary(p1)$df.residual),
            1-pchisq(p_d2, summary(p2)$df.residual),
            1-pchisq(p_d3, summary(p3)$df.residual),
            1-pchisq(p_d4, summary(p4)$df.residual)),
  
  Dispersion_B = c(p_d1/summary(p1)$df.residual,
                   p_d2/ summary(p1)$df.residual,
                   p_d3/ summary(p1)$df.residual,
                   p_d4/ summary(p1)$df.residual),
  
  Deviance_F = c(deviance(p1),
                 deviance(p2),
                 deviance(p3),
                 deviance(p4)),
  
  GOF_F = c(1-pchisq(deviance(p1), summary(p1)$df.residual),
            1-pchisq(deviance(p2), summary(p2)$df.residual),
            1-pchisq(deviance(p3), summary(p3)$df.residual),
            1-pchisq(deviance(p4), summary(p4)$df.residual)),
  
  Dispersion_B = c(deviance(p1)/summary(p1)$df.residual,
                   deviance(p2)/summary(p2)$df.residual,
                   deviance(p3)/summary(p3)$df.residual,
                   deviance(p4)/summary(p4)$df.residual),
  
  row.names = c('Hour', 'Hour_Month', 
                'Hour_Game.Day', 'Hour_Month_Game.Day'))

p_gof


# All eight models failed the deviance goodness-of-fit
# test and showed evidence of being overdispersed. One 
# method of addressing overdispersion is using negative
# binomial regression.


# Bayesian negative binomial regression models adapted from:
# https://georgederpa.github.io/teaching/countModels.html
# 
# Accessed on: November 24, 2020
#
#
# Formulas for deviance residuals, log-likelihood, and BIC for
# the negative binomial distribution adapated from formulas
# listed in the following PDF:
# https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Negative_Binomial_Regression.pdf
#
# Accessed on: November 29, 2020


# Negative binomial models and priors
nb_jags = '
model{
  for(i in 1:n){
  
    # LIKELIHOODS
  
    # Model 1 (Count~Hour)
    Y1[i] ~ dnegbin(p1[i], r1)
    log(lambda1[i]) <- inprod(X1[i,], beta1[])
    p1[i] <- r1/(r1+lambda1[i])
	  
	  # Model 2 (Count~Hour+Month)
	  Y2[i] ~ dnegbin(p2[i], r2)
    log(lambda2[i]) <- inprod(X2[i,], beta2[])
    p2[i] <- r2/(r2+lambda2[i])
	  
	  # Model 3 (Count~Hour+Game.Day)
	  Y3[i] ~ dnegbin(p3[i], r3)
    log(lambda3[i]) <- inprod(X3[i,], beta3[])
    p3[i] <- r3/(r3+lambda3[i])
	  
	  # Model 4 (Count~Hour+Month+Game.Day)
	  Y4[i] ~ dnegbin(p4[i], r4)
    log(lambda4[i]) <- inprod(X4[i,], beta4[])
    p4[i] <- r4/(r4+lambda4[i])
  }
  
  
  
  # PRIORS 
  
  # Priors for Model 1
  r1 ~ dunif(1, 100)
  for(i in 1:12){
    beta1[i] ~ dnorm(0, 0.001)
  }
  
  # Priors for Model 2
  r2 ~ dunif(1, 100)
  for(i in 1:15){
    beta2[i] ~ dnorm(0, 0.001)
  }
  
  # Priors for Model 3
  r3 ~ dunif(1, 100)
  for(i in 1:13){
    beta3[i] ~ dnorm(0, 0.001)
  }
  
  # Priors for Model 4
  r4 ~ dunif(1, 100)
  for(i in 1:16){
    beta4[i] ~ dnorm(0, 0.001)
  }

	
	
	
  # PREDICTIONS USING THE TEST SET
  
	for(i in 1:m){
	
	  # Model 1 Predictions
	  log(lambda1_star[i]) <- inprod(X1_test[i,], beta1[])
		p1_star[i] <- r1/(r1+lambda1_star[i])
		pred1[i] ~ dnegbin(p1_star[i], r1)
		
		# Model 2 Predictions
		log(lambda2_star[i]) <- inprod(X2_test[i,], beta2[])
		p2_star[i] <- r2/(r2+lambda2_star[i])
		pred2[i] ~ dnegbin(p2_star[i], r2)
		
		## Model 3 Predictions
		log(lambda3_star[i]) <- inprod(X3_test[i,], beta3[])
		p3_star[i] <- r3/(r3+lambda3_star[i])
		pred3[i] ~ dnegbin(p3_star[i], r3)
		
		# Model 4 Predictions
		log(lambda4_star[i]) <- inprod(X4_test[i,], beta4[])
		p4_star[i] <- r4/(r4+lambda4_star[i])
		pred4[i] ~ dnegbin(p4_star[i], r4)
	}
}
'


# Data
nb_data = list(n=nrow(train_X),
               m=nrow(test_X),
               
               X1=train_X[,1:12],
               X2=train_X[,1:15],
               X3=train_X[,c(1:12, 16)],
               X4=train_X,
               
               X1_test=test_X[,1:12],
               X2_test=test_X[,1:15],
               X3_test=test_X[,c(1:12, 16)],
               X4_test=test_X,
               
               Y1=train_Y,
               Y2=train_Y,
               Y3=train_Y,
               Y4=train_Y)


# Initialize models
nb_models = jags.model(file=textConnection(nb_jags), data=nb_data,
                       inits=list(.RNG.name = 'base::Wichmann-Hill',
                                  .RNG.seed =7))


# Burn in
update(nb_models, n.iter=1000)


# Sample
nb_outputs = coda.samples(nb_models,
                          variable.names = c('beta1', 'beta2', 
                                             'beta3', 'beta4',
                                             'lambda1', 'lambda2',
                                             'lambda3', 'lambda4',
                                             'pred1', 'pred2', 
                                             'pred3', 'pred4',
                                             'r1', 'r2', 
                                             'r3', 'r4'),
                          n.iter = iter)


# Create 4 frequentist models
nb1 = glm.nb(Count~Hour, data=train_data)
nb2 = glm.nb(Count~Hour+Month, data=train_data)
nb3 = glm.nb(Count~Hour+Game.Day, data=train_data)
nb4 = glm.nb(Count~Hour+Month+Game.Day, data=train_data)


# Extract estimate of each Bayesian node
nb_est = summary(nb_outputs)$statistics[,1]


# Extract Bayesian 95% equi-tailed credible sets
nb_quants = summary(nb_outputs)$quantiles[,c(1,5)]


# Create a table of coefficients with their credible set
nb_beta1 = cbind(Estimate=nb_est[1:12], nb_quants[1:12,])

nb_beta2 = cbind(Estimate=nb_est[13:27], nb_quants[13:27,])

nb_beta3 = cbind(Estimate=nb_est[28:40], nb_quants[28:40,])

nb_beta4 = cbind(nb_est[41:56], nb_quants[41:56,])


# Rename row names for better interpretation
rownames(nb_beta1) = c('(Intercept)', 'Hour13', 'Hour14', 'Hour15', 'Hour16', 
                       'Hour17', 'Hour18', 'Hour19', 'Hour20', 'Hour21', 
                       'Hour22', 'Hour23')

rownames(nb_beta2) = c('(Intercept)', 'Hour13', 'Hour14', 'Hour15', 'Hour16',
                       'Hour17', 'Hour18', 'Hour19', 'Hour20', 'Hour21', 
                       'Hour22', 'Hour23', 'Month10', 'Month11', 'Month12')

rownames(nb_beta3) = c('(Intercept)', 'Hour13', 'Hour14', 'Hour15', 'Hour16',
                       'Hour17', 'Hour18', 'Hour19', 'Hour20', 'Hour21', 
                       'Hour22', 'Hour23', 'Game.Day')

rownames(nb_beta4) = c('(Intercept)', 'Hour13', 'Hour14', 'Hour15', 'Hour16',
                       'Hour17', 'Hour18', 'Hour19', 'Hour20', 'Hour21', 
                       'Hour22', 'Hour23', 'Month10', 'Month11', 'Month12', 
                       'Game.Day')

nb_beta1
nb_beta2
nb_beta3
nb_beta4


# Print GLM coefficients with 95% confidence intervals
cbind(Estimate=coef(nb1), confint(nb1))
cbind(Estimate=coef(nb2), confint(nb2))
cbind(Estimate=coef(nb3), confint(nb3))
cbind(Estimate=coef(nb4), confint(nb4))


# Coefficients should match the associated Poisson model
# (or be approximately the same for the Bayesian models 
# due to the MCMC sampling). However, the variance assumption 
# of Poisson regression is now loosened allowing for a better
# fit.


# Extract fitted values for Bayesian models
nb_fv1 = nb_est[57:564]
nb_fv2 = nb_est[565:1072]
nb_fv3 = nb_est[1073:1580]
nb_fv4 = nb_est[1581:2088]


# Extract r for each NB(p,r) distribution
r1 = nb_est[2601]
r2 = nb_est[2602]
r3 = nb_est[2603]
r4 = nb_est[2404]


# Negative binomial, NB(p,r), deviance residuals  
# d_i=sign(Y_i-\lambda_i)\sqrt{2[Y_i\log(Y_i/\lambda_i)
# -(Y_i+r)\log(\frac{1+Y_i/r}{1+\lambda_i/r})]}
  

# Calculate deviance residuals
nb_dev_res = function(fv, r){
  
  dr = sign(train_Y-fv)*sqrt(2*(train_Y*log(train_Y/fv)
                                -(train_Y+r)*log((1+train_Y/r)/(1+fv/r))))
  
  return(dr)
}


# Calculate deviance residuals for Bayesian models
nb_dr = data.frame(
  nb_dr1 = nb_dev_res(nb_fv1, r1),
  nb_dr2 = nb_dev_res(nb_fv2, r2),
  nb_dr3 = nb_dev_res(nb_fv3, r3),
  nb_dr4 = nb_dev_res(nb_fv4, r4),
  row.names = c(1:508))


# Residual analysis
# Residuals vs fitted values
ggarrange(
  ggplot(nb_dr, aes(x=nb_fv1, y=nb_dr1))+geom_point()+
    xlab('Fitted Values')+ylab('Residuals')+
    ggtitle('Bayesian Model 1'),
  
  ggplot(nb_dr, aes(x=nb_fv2, y=nb_dr2))+geom_point()+
    xlab('Fitted Values')+ylab('Residuals')+
    ggtitle('Bayesian Model 2'),
  
  ggplot(nb_dr, aes(x=nb_fv3, y=nb_dr3))+geom_point()+
    xlab('Fitted Values')+ylab('Residuals')+
    ggtitle('Bayesian Model 3'),
  
  ggplot(nb_dr, aes(x=nb_fv4, y=nb_dr4))+geom_point()+
    xlab('Fitted Values')+ylab('Residuals')+
    ggtitle('Bayesian Model 4'))


ggarrange(
  ggplot(nb1, aes(nb1$fitted.values, nb1$residuals))+
    geom_point()+xlab('Fitted Values')+
    ylab('Residuals')+ggtitle('Frequentist Model 1'),
  
  ggplot(nb2, aes(nb2$fitted.values, nb2$residuals))+
    geom_point()+xlab('Fitted Values')+
    ylab('Residuals')+ggtitle('Frequentist Model 2'),
  
  ggplot(nb3, aes(nb3$fitted.values, nb3$residuals))+
    geom_point()+xlab('Fitted Values')+
    ylab('Residuals')+ggtitle('Frequentist Model 3'),
  
  ggplot(nb4, aes(nb4$fitted.values, nb4$residuals))+
    geom_point()+xlab('Fitted Values')+
    ylab('Residuals')+ggtitle('Frequentist Model 4'))


# Histogram of the Bayesian residuals
ggarrange(
  ggplot(nb_dr, aes(x=nb_dr1))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Bayesian Model 1'),
  
  ggplot(nb_dr, aes(x=nb_dr2))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Bayesian Model 2'),
  
  ggplot(nb_dr, aes(x=nb_dr3))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Bayesian Model 3'),
  
  ggplot(nb_dr, aes(x=nb_dr4))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Bayesian Model 4'))


# Histogram of the frequentist residuals
ggarrange(
  ggplot(nb1, aes(x=nb1$residuals))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Frequentist Model 1'),
  
  ggplot(nb2, aes(x=nb2$residuals))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Frequentist Model 2'),
  
  ggplot(nb3, aes(x=nb3$residuals))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Frequentist Model 3'),
  
  ggplot(nb4, aes(x=nb4$residuals))+geom_histogram()+
    xlab('Deviance Residuals')+ggtitle('Frequentist Model 4'))


# QQ plot of the Bayesian residuals
ggarrange(ggplot(nb_dr, aes(sample=nb_dr1))+stat_qq()+stat_qq_line()+
            ggtitle('Bayesian Model 1'),
          
          ggplot(nb_dr, aes(sample=nb_dr2))+stat_qq()+stat_qq_line()+
            ggtitle('Bayesian Model 2'),
          
          ggplot(nb_dr, aes(sample=nb_dr3))+stat_qq()+stat_qq_line()+
            ggtitle('Bayesian Model 3'),
          
          ggplot(nb_dr, aes(sample=nb_dr4))+stat_qq()+stat_qq_line()+
            ggtitle('Bayesian Model 4'))


# QQ plot of the frequentist residuals
ggarrange(ggplot(nb1, aes(sample=nb1$residuals))+stat_qq()+stat_qq_line()+
            ggtitle('Frequentist Model 1'),
          
          ggplot(nb2, aes(sample=nb2$residuals))+stat_qq()+stat_qq_line()+
            ggtitle('Frequentist Model 2'),
          
          ggplot(nb3, aes(sample=nb3$residuals))+stat_qq()+stat_qq_line()+
            ggtitle('Frequentist Model 3'),
          
          ggplot(nb4, aes(sample=nb4$residuals))+stat_qq()+stat_qq_line()+
            ggtitle('Frequentist Model 4'))


# Log-likelihood  
# LL_i = \log(\Gamma(Y_i+r))-\log(\Gamma(r))-\log(\Gamma(Y_i+1))
# -r\log(1+\lambda_i/r)-Y_i\log(1+\lambda_i/r)+Y_i\log(1/r))
# +Y_i\log(\lambda_i)
  
# BIC  
# BIC = -2\sum_{i=1}^n LL_i + p\log(n)$, where $p$=number of 
# predicting variables.


# Calculate BIC
bayes_bic = function(fv, num_params, r){
  
  ll = log(gamma(train_Y+r))-log(gamma(r))-log(gamma(train_Y+1))-
    r*log(1+fv/r)-train_Y*log(1+fv/r)+train_Y*log(1/r)+
    train_Y*log(fv)
  
  return(-2*sum(ll)+num_params*log(508))
}


# Table of BIC values
bic = data.frame(
  BIC_B = c(bayes_bic(nb_fv1, 12, r1),
            bayes_bic(nb_fv2, 15, r2),
            bayes_bic(nb_fv3, 13, r3),
            bayes_bic(nb_fv4, 16, r4)),
  
  BIC_F = c(AIC(nb1, k=log(508)),
            AIC(nb2, k=log(508)),
            AIC(nb3, k=log(508)),
            AIC(nb4, k=log(508))),
  
  row.names = c('Hour', 'Hour_Month', 
                'Hour_Game.Day', 'Hour_Month_Game.Day'))

bic


# Calculate the deviance of each Bayesian model
nb_d1 = sum(nb_dr$nb_dr1^2)
nb_d2 = sum(nb_dr$nb_dr2^2)
nb_d3 = sum(nb_dr$nb_dr3^2)
nb_d4 = sum(nb_dr$nb_dr4^2)


# Summary table (deviance, p-value, dispersion)
nb_gof = data.frame(
  Deviance_B = c(nb_d1, nb_d2, nb_d3, nb_d4),
  
  GOF_B = c(1-pchisq(nb_d1, summary(nb1)$df.residual),
            1-pchisq(nb_d2, summary(nb2)$df.residual),
            1-pchisq(nb_d3, summary(nb3)$df.residual),
            1-pchisq(nb_d4, summary(nb4)$df.residual)),
  
  Deviance_F = c(deviance(nb1),
                 deviance(nb2), 
                 deviance(nb3), 
                 deviance(nb4)),
  
  GOF_F = c(1-pchisq(deviance(nb1), summary(nb1)$df.residual),
            1-pchisq(deviance(nb2), summary(nb2)$df.residual),
            1-pchisq(deviance(nb3), summary(nb3)$df.residual),
            1-pchisq(deviance(nb4), summary(nb4)$df.residual)),
  
  row.names = c('Hour', 'Hour_Month', 
                'Hour_Game.Day', 'Hour_Month_Game.Day'))

nb_gof


# Bayesian predictions
nb_pred1 = nb_est[2089:2216]
nb_pred2 = nb_est[2217:2344]
nb_pred3 = nb_est[2345:2472]
nb_pred4 = nb_est[2473:2600]


# GLM predictions
nb1_pred = predict(nb1, test_data, type='response')
nb2_pred = predict(nb2, test_data, type='response')
nb3_pred = predict(nb3, test_data, type='response')
nb4_pred = predict(nb4, test_data, type='response')


# Mean Square Prediction Error and Precision Error
mspe = function(pred){
  mspe = mean((pred-test_data$Count)^2)
  return(mspe)
}

precision = function(pred){
  precision = sum((pred-test_data$Count)^2)/
    sum((test_data$Count-mean(test_data$Count))^2)
  
  return(precision)
}

nb_errors = data.frame(
  
  MSPE_B = c(mspe(nb_pred1), mspe(nb_pred2), 
             mspe(nb_pred3), mspe(nb_pred4)),
  
  MSPE_F = c(mspe(nb1_pred), mspe(nb2_pred), 
             mspe(nb3_pred), mspe(nb4_pred)),
  
  Precison_B = c(precision(nb_pred1), precision(nb_pred2),
                 precision(nb_pred3), precision(nb_pred4)),
  
  Precison_F = c(precision(nb1_pred), precision(nb2_pred),
                 precision(nb3_pred), precision(nb4_pred)),
  
  row.names = c('Hour', 'Hour_Month', 
                'Hour_Game.Day', 'Hour_Month_Game.Day'))

nb_errors
