context("Pattern generation")
library(rveg)
setwd(file.path("..",".."))

test_that("we can load config from json", {
    configFilename <- file.path("..","testdata","patternGenConfig.json")
    config <- loadConfig(configFilename)
    expect_gt(config$DeltaX, 0)
    expect_gt(config$DeltaY, 0)

})


test_that("We can load a starting pattern", {
    startingPatternFile <- file.path("..","testdata","random-initialisation_m=50.csv")
    pattern <- getStartingPattern(startingPatternFile)
    expect_equal(dim(pattern),c(50,50))
    expect_equal(min(pattern), 0)
    expect_equal(max(pattern), 90)
})


test_that("We can binarize a matrix of floats", {
    patternFile <- file.path("..","testdata","test_float_pattern_50x50_1.csv")
    pattern <- as.matrix(read.csv(patternFile, header=FALSE))
    binaryPattern <- binarizePattern(pattern)
    expect_equal(dim(binaryPattern),c(50,50))
    expect_equal(max(binaryPattern),1)
    expect_equal(min(binaryPattern),0)
    expect_equal(sum(binaryPattern), 1055)
    ## if we set threshold to zero we should get all 1s.
    binaryPattern <- binarizePattern(pattern, 0)
    expect_equal(sum(binaryPattern), 2500)
    ## if we set threshold high we should get all 0s.
    binaryPattern <- binarizePattern(pattern, max(pattern)+1.)
    expect_equal(sum(binaryPattern), 0)
})


test_that("We generate a known pattern from given input", {
    print(getwd())
    genConfig <- file.path("..","testdata","patternGenConfig.json")
    startingPatternFile <- file.path("..","testdata","random-initialisation_m=50.csv")
    pattern <- generatePattern(genConfig, startingPatternFile)
    expect_equal(dim(pattern),c(50,50))
    expect_gt(max(pattern),0)
    expect_lt(min(pattern),1)
})
