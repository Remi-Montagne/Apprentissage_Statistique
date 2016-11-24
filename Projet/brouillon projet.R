############################################
#     Projet Apprentissage Statistique     #
############################################

rm(list=ls())

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
library(tree)
library(glmnet)
library(randomForest)
library(MASS)
library(ROCR)

pdf(file = "figures.pdf")
#################################
#    Observation des données    #
#################################

### observation de correlation entre variables

for(i in c(2, 12, 22)){
  pairs(tab2[,i:(i+9)])
}

round(cor(tab2[,2:col]), digits = 2)
correl <- cor(tab2[,-1])
heatmap.2(correl, sepcolor = "black", trace = "none")

### boxplots : distribution des variable en fonction de la variable réponse
noms <- colnames(tab2)
par(mfrow = c(1,1))
for(i in 2:col) {
  boxplot(tab2[,i]~tab2$Y,xlab = toString(i+1), main = noms[i]) 
}


#################################
#      Generation des jeux      #
# d'entrainement et de test     #
#################################

set.seed(123) #afin d'obtenir des resultats reproductibles, la graine est fixee, ici a 123
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
model_total <- glm(Y~., data = train[,1:22], family = "binomial")
summary(model_total)
# L'algorithme n'a pas convergé

### Construction d'un modèle pas à pas à partir du modèle nul en minimisant le critere AIC
model_glm <- step(model_min,scope=list(lower=model_min, upper = model_total),direction="both")
summary(model_glm)

### Validation croisee

k <- 5
set.seed((123))
group <- sample(1:k,length(tab2[,1]), replace = T)
tab_pour_cv <- cbind(tab2,group)
seuils <- seq(0, 1, 0.1)
s_min = 0
best_pred_err <- 1

plot("s", "errer", type = "n", xlim = c(0,1), ylim = c(0,0.2))

for(s in seuils)
{
  
  list_mal_classes <- c()
  
  for(i in 1:k)
  {
    train_cv <-tab2[tab_pour_cv[,col+1] != i,1:22]
    test_cv <-tab2[tab_pour_cv[,col+1] == i,1:22]
    
    model_tot <- glm(Y~., data = train_cv, family = "binomial")
    set.seed((123))
    model_glm <- step(model_min,scope=list(lower=model_min, upper = model_tot),direction="both")
    
    glm_predit = rep("B", length(test_cv[,1])) #on va mettre nos predictions (M ou B) dedans
    probs = predict(model_glm, newdata = test_cv[,-1], type = "response") #prediction de proba
    glm_predit[probs > s] <- "M" #si on est au-dessus de 0.5, on met M dans glm-predit. Sinon on laisse les B, qui y sont déjà.
    
    list_mal_classes <- c(list_mal_classes, mean(test_cv$Y!=glm_predit)) #moyenne des mal classe
  }
  
  points(s, mean(list_mal_classes))
  
  if(mean(list_mal_classes) < best_pred_err)
  {
    s_min <- s
    best_pred_err <- mean(list_mal_classes)
  }
}

s_min
best_pred_err


### Creation du modele a comparer avec les autres methodes

#modele
model_total <- glm(Y~., data = train[1:22], family = "binomial")
set.seed((123))
model_glm <- step(model_min,scope=list(lower=model_min, upper = model_total),direction="both")

#prediction
glm_predit = rep("B", length(test[,1])) #on va mettre nos predictions (M ou B) dedans
probs = predict(model_glm, newdata = test, type = "response") #prediction de proba
glm_predit[probs > s_min] <- 'M' #si on est au-dessus de 0.5, on met M dans glm-predit. Sinon on laisse les B, qui y sont déjà.

table(test$Y, glm_predit)
error_glm <- mean(test$Y!=glm_predit)
error_glm

#################################
#           Arbre CART          #
#################################

### Obtention du modele
model_tree <- tree(Y~., data = train)
summary(model_tree)
plot(model_tree)
text(model_tree, cex=0.7)

### Raffinement du modele : elagage

# etude d'elagages possibles
set.seed(123)
CV_tree = cv.tree(model_tree, FUN = prune.misclass) #remarque : par defaut, on a FUN=prune.tree pas bon ça correspond a l'erreur d'apprentissage
CV_tree
plot(CV_tree$size, CV_tree$dev, type = 'b')

# selection du plus petit nombre de feuilles donnant la plus petite erreur de prediction
nb_feuille_candidat <- which(CV_tree$dev == min(CV_tree$dev)) 
feuilles <- CV_tree$size[nb_feuille_candidat]
feuilles_min <- min(feuilles)

# nouvel arbre
model_prune <- prune.misclass(model_tree, best=feuilles_min)
plot(model_prune)
text(model_prune, cex=0.7)

### Erreur test de l'arbre
tree_predit=predict(model_prune, newdata=test, type="class")
error_tree <- mean(tree_predit != test$Y)


#################################
#      Regression Ridge         #
#################################

### Obtention d'un premier modele
xtrain=model.matrix(Y~., train)[,-1] #passage de la dataframe en matrice
ytrain = train$Y
xtest=model.matrix(Y~., test)[,-1]

model_ridge <- glmnet(xtrain,ytrain, family = "binomial", alpha = 0)

### Choix du lambda par validation croisee

# 1e test
set.seed(123)
cv.out= cv.glmnet(xtrain, ytrain, family = "binomial", alpha = 0)
plot(cv.out) #trace cvm en fonction de lambda

# minimisation du lambda
lambda_seq = seq(0.0001, 0.01, 0.0001)
set.seed(123)
cv.out= cv.glmnet(xtrain, ytrain, family = "binomial", alpha = 0, lambda = lambda_seq)
plot(cv.out) #trace cvm en fonction de lambda

lambda_opt = cv.out$lambda.min
lambda_opt

### Prédiction d'un modele avec le lambda optimal
model_ridge = glmnet(xtrain, ytrain, family = "binomial", alpha = 0, lambda = lambda_seq)

### Erreur test du modele Ridge
ridge_predit <- predict(object = model_ridge, newx = xtest, s = lambda_opt, type = "class")
table(test$Y, ridge_predit)
error_ridge <- mean(test$Y != ridge_predit)


#################################
#        Random Forest          #
#################################

### Construction d'une foret aleatoire
set.seed(123)
model_rf <- randomForest(Y~., data = train)
print(model_rf) #affiche un summary de la foret

### Erreur test
rf_predit=predict(model_rf, newdata=test, type="class")
table(test$Y, rf_predit)
error_rf <- mean(test$Y != rf_predit)


#################################
#   Comparaison des erreur et   #
#  construction du model final  #
#################################

### Taux de mal classes
cat("arbre CART :", error_glm, "\n", "arbre CART :", error_tree, "\n","ridge :", error_ridge, "\n","random forest :", error_rf, "\n\n\n")

### Matrices de confusion, FPR (qui correspond ici a la detection des malades)

#regression logistique
mc_glm <- table(test$Y, glm_predit)
mc_glm
Se_glm <- mc_glm[2,2]/(mc_glm[2,1] + mc_glm[2,2])

#arbre CART
mc_tree <- table(test$Y, tree_predit)
mc_tree
Se_tree <- mc_tree[2,2]/(mc_tree[2,1] + mc_tree[2,2])

#methode ridge
mc_ridge <- table(test$Y, ridge_predit)
mc_ridge
Se_ridge <- mc_ridge[2,2]/(mc_ridge[2,1] + mc_ridge[2,2])

#methode rf
mc_rf <- table(test$Y, rf_predit)
mc_rf
Se_rf <- mc_rf[2,2]/(mc_rf[2,1] + mc_rf[2,2])

cat("glm :", Se_glm, "\n", "arbre CART :", Se_tree, "\n","ridge :", Se_ridge, "\n","random forest :", Se_rf, "\n\n\n")

### Coursbes ROC

#pour l'arbre CART
pred_tree = predict(model_tree, newdata = test)
predic_tree = prediction(pred_tree[,2], test$Y)
perf_tree <- performance(predic_tree, "tpr", "fpr")
plot(perf_tree, main = "Courbes ROC")

#pour la regression logistique
pred_glm = predict(model_glm, newdata = test, type = "response")
predic_glm = prediction(pred_glm, test$Y)
perf_glm <- performance(predic_glm, "tpr", "fpr")
plot(perf_glm, col = 4, add = TRUE)

#pour la methode ridge
pred_ridge = predict(model_ridge, newx = xtest, s = lambda_opt, type = "response")
predic_ridge = prediction(pred_ridge[,1], test$Y)
perf_ridge <- performance(predic_ridge, "tpr", "fpr")
plot(perf_ridge, col = 2, add = TRUE)

#pour la random forest
pred_rf = predict(model_rf, newdata = test, type = "prob")
predic_rf = prediction(pred_rf[,2], test$Y)
perf_rf <- performance(predic_rf, "tpr", "fpr")
plot(perf_rf, col = 3, add = TRUE)

abline(a = 0, b = 1, col = 4)


### AUC
auc_glm <- performance(predic_glm, "auc")
auc_tree <- performance(predic_tree, "auc")
auc_ridge <- performance(predic_ridge, "auc")
auc_rf <- performance(predic_rf, "auc")

cat("glm :", as.numeric(auc_glm@y.values) , "\n", "arbre CART :", as.numeric(auc_tree@y.values), "\n","ridge :", as.numeric(auc_ridge@y.values), "\n","random forest :", as.numeric(auc_rf@y.values), "\n\n\n")

dev.off()