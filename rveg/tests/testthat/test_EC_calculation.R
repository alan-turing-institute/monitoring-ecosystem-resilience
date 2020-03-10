context("Subgraph centrality/Euler Characteristic calculation")
library(rveg)
setwd(file.path("..",".."))

test_that("we can load pattern from csv", {

    filename <- file.path("inst","extdata","binary_labyrinths_50.csv")
    pattern <- read_from_csv(filename)
    expect_equal(dim(pattern),c(50,50))
    expect_equal(sum(pattern), 1333)
})


test_that("we can load pattern from png", {
    filename <- file.path("inst","extdata","black_and_white_diagonal.png")
    pattern <- read_from_png(filename)
    expect_equal(dim(pattern),c(50,50))
    expect_equal(sum(pattern), 1275)
})


test_that("we can make edges from a pattern", {
    filename <- file.path("inst","extdata","binary_labyrinths_50.csv")
    pattern <- read_from_csv(filename)
    edges <- make_edges(pattern)
    expect_equal(dim(edges),c(7810,2))
    expect_equal(edges[1,], c(801,851))
    expect_equal(edges[7810,], c(2450,2400))
})

test_that("We can calculate a feature vector", {
    filename <- file.path("inst","extdata","binary_labyrinths_50.csv")
    pattern <- read_from_csv(filename)
    feature_vec <- calc_EC(pattern)
    expect_equal(length(feature_vec), 20)
    expect_equal(feature_vec[[1]], -22)
    expect_equal(feature_vec[[20]], -2572)

})
