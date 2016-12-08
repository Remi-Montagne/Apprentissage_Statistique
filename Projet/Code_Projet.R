############################################
#     Projet Apprentissage Statistique     #
############################################

rm(list=ls())
par(mfrow = c(1,1))

#################################
# Chargement data et librairies #
#################################

### chargement dee données
tab <- read.table(file = "wdbc.data", header = FALSE, sep = ',')
names(tab)[2]="Y" #la colonne 2 est la variable reponse, renommee Y
names(tab)[1]="Id" #la colonne 1 est l'identifiant, renommé Id
tab2 <- tab[,-1] # data2 reprend la même table mais sans la colonne des identifiants
#tab_mean <- tab[,2:12] #on selectionne uniquement les valeur moyennes des 10 criteres etudies

lin <- nrow(tab2) #nombre de lignes
lin
col <- ncol(tab2) #nombre de lignes

### chargement des librairies
library(gplots)
library(boot)
library(glmnet)
library(randomForest)
library(MASS)
library(ROCR)

#pdf(file = "figures.pdf")
#################################
#    Observation des données    #
#################################

### observation de correlation entre variables

for(i in c(2, 12, 22)){
  pairs(tab2[,i:(i+9)])
}

round(cor(tab2[,2:col]), digits = 2)
correl <- cor(tab2[,-1])
h <- hclust(correl)
heatmap(correl, sepcolor = "black", trace = "none", main = "Corrélation entre covaraibles")

for(i in 2:11) {
  print(round(cor(tab2[,c(i, (i+10), (i+20))]), digits = 2))
  pairs(tab2[,c(i, (i+10), (i+20))], panel = panel.smooth, print("Corrélation entre covariables"))
  
}

### boxplots : distribution des variable en fonction de la variable réponse
noms <- colnames(tab2)
par(mfrow = c(1,1))
for(i in 2:col) {
  boxplot(tab2[,i]~tab2$Y,xlab = toString(i+1), main = "")
  legend("top", legend = paste("Répartition des valeurs de la variable ", noms[i]), cex = 0.8, lty = 0)
  hist(tab2[,i], main = "")
  legend("topright", legend = paste("Répartition des valeurs de la variable ", noms[i]), cex = 0.8, bty = "n")
}


#################################
#      Generation des jeux      #
# d'entrainement et de test     #
#################################

set.seed(1234) #afin d'obtenir des resultats reproductibles, la graine est fixee, ici a 123
test = sample(1:lin, round(lin/3))
train = -test
train = tab2[train,]
test = tab2[test,]


#verification que les proportions de chaque type de patient sont similare dans les differents jeux de donnees
cat("proportion de B dans le tableau total : ", mean(tab2[,1] == "B"))
cat("proportion de B dans train : ", mean(train[,1] == "B"))
cat("proportion de B dans test : ", mean(test[,1] == "B"))


#################################
#     Regression logistique     #
#################################

### Essai de génération d'un model a partir de toutes les covariables : model_total
model_min <- glm(Y~1, data = train, family = "binomial")
model_total <- glm(Y~., data = train, family = "binomial", maxit = 50)
summary(model_total)
# L'algorithme n'a pas convergé

### Construction d'un modèle pas à pas à partir du modèle nul en minimisant le critere AIC
model_asc <- step(model_min,scope=list(lower=model_min, upper = model_total),direction="both")
summary(model_asc)

### Construction d'un modèle pas à pas à partir du modèle nul en minimisant le critere AIC
model_desc <- step(model_total,scope=list(lower=model_min, upper = model_total),direction="both")
summary(model_desc)

anova(model_asc, model_total, test = "Chisq") #Verification que l'on ne rejette pas le model_glm
anova(model_desc, model_total, test = "Chisq") #Verification que l'on ne rejette pas le model_glm

### Validation croisee
k <- 5
set.seed((1234))
group <- sample(1:k,length(train[,1]), replace = T)
tab_pour_cv <- cbind(train,group)
seuils <- seq(0, 1, 0.025)

plot("s", "errer", type = "n", xlim = c(0,1), ylim = c(0,0.1))


noms <- list(seq(1,5), seuils)
errors <- matrix(nrow = k, ncol = length(seuils), dimnames = noms)

for(i in 1:k) {
    train_cv <-tab2[tab_pour_cv[,col+1] != i,]
    test_cv <-tab2[tab_pour_cv[,col+1] == i,]
    
    model_tot <- glm(Y~., data = train_cv, family = "binomial")
    model_glm <- step(model_min,scope=list(lower=model_min, upper = model_tot),direction="both")
    
    j <- 1
    for(s in seuils){
    glm_predit = rep("B", length(test_cv[,1])) #on va mettre nos predictions (M ou B) dedans
    probs = predict(model_glm, newdata = test_cv[,-1], type = "response") #prediction de proba
    glm_predit[probs > s] <- "M" #si on est au-dessus de 0.5, on met M dans glm-predit. Sinon on laisse les B, qui y sont déjà.
    
    errors[i, j] <- mean(test_cv$Y!=glm_predit) #moyenne des mal classe
    j <- j+1
    }
}
#errors <- data.frame(errors)
err_moyennes <- apply(errors, 2, mean)
plot(seuils, err_moyennes, ylim = c(0,0.15))

s_min <- noms[[2]][which.min(err_moyennes)]
abline(v = s_min, lty = 2)
s_min
best_pred_err = min(err_moyennes)
best_pred_err

### Creation du modele a comparer avec les autres methodes
#modele
model_total <- glm(Y~., data = train, family = "binomial")
set.seed((1234))
model_glm <- step(model_min,scope=list(lower=model_min, upper = model_total),direction="both")
summary(model_glm)

#prediction
glm_predit = rep("B", length(test[,1])) #on va mettre nos predictions (M ou B) dedans
probs = predict(model_glm, newdata = test, type = "response") #prediction de proba
glm_predit[probs > s_min] <- 'M' #si on est au-dessus de 0.5, on met M dans glm-predit. Sinon on laisse les B, qui y sont déjà.

table(test$Y, glm_predit)
error_glm <- mean(test$Y!=glm_predit)


#################################
#      Regression Ridge         #
#################################

### Premier modele avec toutes les données d'entraînement

# Obtention du modèle
x=model.matrix(Y~., tab2)[,-1] #passage de la dataframe en matrice
y=tab2$Y
xtrain=model.matrix(Y~., train)[,-1] #passage de la dataframe en matrice
ytrain = train$Y
xtest=model.matrix(Y~., test)[,-1]

model_ridge <- glmnet(xtrain,ytrain, family = "binomial", alpha = 0)

# Choix du lambda par validation croisee
# 1e test
set.seed(1234)
cv.out= cv.glmnet(xtrain, ytrain, family = "binomial", alpha = 0)
plot(cv.out) #trace cvm en fonction de lambda

# minimisation du lambda
lambda_seq = seq(0.0001, 0.02, 0.0001)
set.seed(1234)
cv.out= cv.glmnet(xtrain, ytrain, family = "binomial", alpha = 0, lambda = lambda_seq)
plot(cv.out) #trace cvm en fonction de lambda

lambda_opt = cv.out$lambda.min
lambda_opt

# Prédiction d'un modele avec le lambda optimal
model_ridge = glmnet(xtrain, ytrain, family = "binomial", alpha = 0, lambda = lambda_seq)

# Erreur test du modele Ridge
ridge_predit <- predict(object = model_ridge, newx = xtest, s = lambda_opt, type = "class")
table(test$Y, ridge_predit)
error_ridge <- mean(test$Y != ridge_predit)



#################################
#        Random Forest          #
#################################

### Construction d'une foret aleatoire
set.seed(1234)
model_rf <- randomForest(Y~., data = train, ntree = 1000)
plot(model_rf)

plot(1,1, type = "n", xlim = c(0,1000), ylim = c(0.0, 0.1))
txt <- c()
color <- c()
for(i in 1:10) {
  set.seed(1234)
  model_rf <- randomForest(Y~., data = train, ntree = 1000, mtry = i)
  oob <- model_rf$err.rate
  points(1:1000, oob[,1], type = "l", add = TRUE, col = i) #on affiche le taux d'erreur pour chaque taille du nombre B d'arbre.
  txt = c(txt, i)
  color = c(color, i)
}
legend("topright", legend = txt, fill = color)

plot(1,1, type = "n", xlim = c(0,1000), ylim = c(0, 0.1))
txt <- c()
color <- c()
for(i in 11:20) {
  set.seed(1234)
  model_rf <- randomForest(Y~., data = train, ntree = 1000, mtry = i)
  oob <- model_rf$err.rate
  points(1:1000, oob[,1], type = "l", add = TRUE, col = i-10) #on affiche le taux d'erreur pour chaque taille du nombre B d'arbre.
  txt = c(txt, i)
  color = c(color, i-10)
}
legend("topright", legend = txt, fill = color)

plot(1,1, type = "n", xlim = c(0,1000), ylim = c(0, 0.1))
txt <- c()
color <- c()
for(i in 21:30) {
  set.seed(1234)
  model_rf <- randomForest(Y~., data = train, ntree = 1000, mtry = i)
  oob <- model_rf$err.rate
  points(1:1000, oob[,1], type = "l", add = TRUE, col = i-20) #on affiche le taux d'erreur pour chaque taille du nombre B d'arbre.
  txt = c(txt, i)
  color = c(color, i-20)
}
legend("topright", legend = txt, fill = color)

top_m = c(5, 6, 7, 11, 23)
plot(1,1, type = "n", xlim = c(0,1000), ylim = c(0.02, 0.1))
txt <- c()
color <- c()
j = 1
for(i in top_m) {
  set.seed(1234)
  model_rf <- randomForest(Y~., data = train, ntree = 1000, mtry = i)
  oob <- model_rf$err.rate
  points(1:1000, oob[,1], type = "l", add = TRUE, col = j) #on affiche le taux d'erreur pour chaque taille du nombre B d'arbre.
  txt = c(txt, i)
  color = c(color, j)
  j = j+1
}
legend("topright", legend = txt, fill = color)

# construction d'une foret aleatoire a partir de 200 arbres et d'un taux m = 6
set.seed(1234)
model_rf <- randomForest(Y~., data = train, mtry = 5, ntree = 200)

### Erreur test
rf_predit=predict(model_rf, newdata=test, type="class")
table(test$Y, rf_predit)
error_rf <- mean(test$Y != rf_predit)


#################################
#   Comparaison des erreur et   #
#  construction du model final  #
#################################

### Taux de mal classes
cat("regression :", error_glm, "\n", "ridge :", error_ridge, "\n","random forest :", error_rf, "\n\n\n")

### Matrices de confusion, FPR (qui correspond ici a la detection des malades)

#regression logistique
mc_glm <- table(test$Y, glm_predit)
mc_glm
Se_glm <- mc_glm[2,2]/(mc_glm[2,1] + mc_glm[2,2])

#methode ridge
mc_ridge <- table(test$Y, ridge_predit)
mc_ridge
Se_ridge <- mc_ridge[2,2]/(mc_ridge[2,1] + mc_ridge[2,2])

#methode rf
mc_rf <- table(test$Y, rf_predit)
mc_rf
Se_rf <- mc_rf[2,2]/(mc_rf[2,1] + mc_rf[2,2])

cat("glm :", Se_glm, "\n","ridge :", Se_ridge, "\n","random forest :", Se_rf, "\n\n\n")

### Coursbes ROC

#pour la regression logistique
pred_glm = predict(model_glm, newdata = test, type = "response")
predic_glm = prediction(pred_glm, test$Y)
perf_glm <- performance(predic_glm, "tpr", "fpr")
plot(perf_glm, col = 1)

#pour la methode ridge
pred_ridge = predict(model_ridge, newx = xtest, s = lambda_opt, type = "response")
predic_ridge = prediction(pred_ridge[,1], test$Y)
perf_ridge <- performance(predic_ridge, "tpr", "fpr")
plot(perf_ridge, col = 2, add = T)

#pour la random forest
pred_rf = predict(model_rf, newdata = test, type = "prob")
predic_rf = prediction(pred_rf[,2], test$Y)
perf_rf <- performance(predic_rf, "tpr", "fpr")
plot(perf_rf, col = 3, add = TRUE)

abline(a = 0, b = 1, col = 4)
legend("right" , legend = c("reg logistique", "ridge", "random forest"), fill = c(1, 2, 3), cex = 0.8)

### AUC
auc_glm <- performance(predic_glm, "auc")
auc_ridge <- performance(predic_ridge, "auc")
auc_rf <- performance(predic_rf, "auc")

cat("glm :", as.numeric(auc_glm@y.values) , "\n","ridge :", as.numeric(auc_ridge@y.values), "\n","random forest :", as.numeric(auc_rf@y.values), "\n\n\n")

#dev.off()