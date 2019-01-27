/*
 * The MIT License
 *
 * Copyright 2019 Rob Garcia at rgarcia@rgprogramming.com.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

package mlalgorithmsinjava;

import static java.lang.System.exit;

/**
 *
 * @author Rob Garcia at rgarcia@rgprogramming.com
 */
public class MLAlgorithmsInJava {

    /** USE DOUBLE WRAPPER, NOT PRIMITIVE TYPE! WE NEED TO DO NULL CHECKS! **/
    
    /**
     * x represents the set of independent variables, known as the set of features.
     * For example, the square footage of a house.
     * x[1][n] must always equal 1 to allow the formula to return theta[0][n] (the y-intercept for all lines of x).
     * There must be a row of x values in the x matrix for each element in the y vector.
     */
    public static Double[][] xMatrix = {{1.0, 1.0},
                                        {1.0, 2.0},
                                        {1.0, 3.0}
    };
    
    /**
     * y represents the dependent variable, known as the output feature.
     * For example, the actual price of a house based on the square footage ($151 per square foot in 2018).
     * There must be an element in the y vector for each row of x values in the x matrix.
     */
    public static Double[] yVector = {1.0, 2.0, 3.0};

    /**
     * theta represents the slope of each point in the set of features.
     * For example, the slope of each home price in relation to it's square footage.
     * The number of elements in each row of theta must match the number elements in each row of x.
     * theta[0][n] is equal to the y-intercept of the prediction.
     */
    public static Double[] thetaVector = {0.0, 0.5};
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        System.out.println("Hello, world!");
        System.out.println("Checking data...");
        // Check that the number of x values equals the number of y values.
        if(xMatrix.length != yVector.length) {
            System.out.println("Check your data: the number of x elements (" + xMatrix.length + ") does not equal the number of y elements (" + yVector.length + ").");
            exit(0);
        }
        // Check that there is a theta value for each value of x (per row). 
        else if(xMatrix[0].length != thetaVector.length) {
            System.out.println("Check your data: the number of x elements (" + xMatrix[0].length + ") does not equal the number of theta elements (" + thetaVector.length + ").");
            exit(0);
        }
        else {
            System.out.println("Data OK.");
            boolean convergenceFlag = false;
            while(convergenceFlag == false) {
                System.out.println("Calculating the number of examples (m)...");
                int m = xMatrix[0].length;
                System.out.println("The number of examples is " + m);
                System.out.println("Calculating the hypotheses: (h\u019F(x) = \u019F\u2080X\u2080 + \u019F\u2081X\u2081 + ... + \u019F\u2099X\u2099)...");
                // Create a vector to hold each element in an x matrix column
                Double[] xVector = new Double[xMatrix[0].length];
                // Create a matrix to hold all the h values per theta and x vector (used to compute sum of squares for each vector)
                Double[] h = new Double[yVector.length];
                Double[] residual = new Double[yVector.length];
                Double[] squaredResidualForCF = new Double[yVector.length];
                Double sumOfSquares = 0.0;
                Double J = 0.0;
                // Must use this instead of Math.pow to avoid "lossy conversion from double to int" error
                Double[] adjustedResidualForGD = new Double[yVector.length];
                Double[] sumOfAdjustedResiduals = new Double[yVector.length];
                Double alpha = 0.0;
                Double[] tempThetaVector = new Double[thetaVector.length];
                /**
                 * To get h, you need to apply each theta vector to each vector of x.
                 * The number of theta vectors controls the outer loop.
                 * Each theta vector must be applied to each x vector,
                 * so the inner loop is controlled by the number of vectors in x.
                 * Each value in a theta vector must be applied to each value in an x vector,
                 * so the innermost loop is controlled by the number of elements
                 * in each theta vector or x vector
                 * (the number of elements in each must be the same for matrix multiplication).
                 **/

                for(int i = 0; i < xMatrix.length; i++) {
                    // Create the vectors for matrix multiplication
                    xVector = xMatrix[i];
                    // Apply the hypothesis function. If it fails, it wil return null and the application will exit.
                    if((h[i] = LinearRegression.Hypothesis(thetaVector, xVector)) == null) {
                        exit(0);
                    }
                    else {
                        // My fun code to print out each hypothesis function! 
                        System.out.printf("h\u019F(x[%d]) = %f or ", (i + 1), h[i]);
                        for(int k = 0; k < xVector.length; k++) {
                            System.out.printf("(%f * %f)", thetaVector[k], xVector[k]);
                            System.out.print(k == (xVector.length - 1) ? "" : " + ");
                        }
                        System.out.println();
                    }
                }
                System.out.println("Calculating the residuals of h\u019F(x) - y...");
                for(int i = 0; i < yVector.length; i++) {
                    residual[i] = h[i] - yVector[i];
                    System.out.printf("h\u019F(x)[%d] - y[%d] = %f or %f - %f\n", i + 1, i + 1, residual[i], h[i], yVector[i]);
                }
                System.out.println("Calculating the squares of the residuals of h\u019F(x) - y...");
                for(int i = 0; i < yVector.length; i++) {
                    squaredResidualForCF[i] = Math.pow((h[i] - yVector[i]), 2);
                    System.out.printf("(h\u019F(x)[%d] - y[%d])\u00B2 = %f or %f\u00B2\n", i + 1, i + 1, squaredResidualForCF[i], residual[i]);
                }
                System.out.println("Calculating the sum of squares of the residuals of h\u019F(x) - y...");
                System.out.print("The sum of squares = ");
                for(int i = 0; i < yVector.length; i++) {
                    sumOfSquares += squaredResidualForCF[i];
                    System.out.printf("%f", squaredResidualForCF[i]);
                    System.out.print(i == (squaredResidualForCF.length - 1) ? " or " : " + ");
                }
                System.out.println(sumOfSquares);
                System.out.println("Calculating the cost function J(\u019F\u2080, \u019F\u2081...\u019F\u2099)) = (1 / (2 * m)) * (\u2211((h(x\u2099) - y\u2099)\u00B2))...");
                J = (1.0 / (2.0 * (m + 1))) * sumOfSquares;
                System.out.printf("J = %f or (1 / (2 * %d)) * %f\n", J, m + 1, sumOfSquares);
                System.out.println("Performing gradient descent...");
                System.out.println("Adjusting original residuals: (h\u019F(x) - y)) * x...");
                for(int i = 0; i < xMatrix.length; i++) {
                    // Initialize adjustedResidualForGD to prevent NullPointerException when using the += assignment operator
                    adjustedResidualForGD[i] = 0.0;
                    for(int j = 0; j < xMatrix[i].length; j++) {
                        adjustedResidualForGD[i] += residual[i] * xMatrix[i][j];
                        System.out.printf("h\u019F(x)[%d] - y[%d] (%f) * x[%d][%d] (%f) = %f (%f)\n", i, i, residual[i], i, j, xMatrix[i][j], (residual[i] * xMatrix[i][j]), adjustedResidualForGD[i]);
                    }
                }
                System.out.println("Summing up adjusted original residuals for each x: (h\u019F(x) - y)) * x...");
                // Initialize sumOfAdjustedResiduals to prevent NullPointerException when using the += assignment operator
                // sumOfAdjustedResiduals = 0.0;
                for(int i = 0; i < xMatrix.length; i++) {
                    if(sumOfAdjustedResiduals[i] == null) {
                        sumOfAdjustedResiduals[i] = 0.0;
                    }
                    sumOfAdjustedResiduals[i] += adjustedResidualForGD[i];
                }
                
                System.out.println();
                convergenceFlag = true;
            }
            /*
            System.out.println("\nCalculating the cost [J(\u019F\u2080, \u019F\u2081...\u019F\u2099)) = 1 / (2 * m) * (\u2211((h(x\u2099) - y\u2099)\u00B2))]...\n");
            Double[] J = new Double[thetaMatrix.length];
            for(int i = 0; i < thetaMatrix.length; i++) {
                if((J[i] = LinearRegression.CostFunction(h[i], yVector)) == null) {
                        exit(0);
                }
                else {
                    // My fun code to print out the results of the cost function! 
                    System.out.print("J(");
                    for(int k2 = 0; k2 < thetaMatrix[0].length; k2++) {
                        System.out.printf("%.2f", thetaMatrix[i][k2]);
                        System.out.print(k2 == (thetaMatrix.length - 1) ? "" : ", ");
                    }
                    System.out.printf(") = %.2f\n", J[i]);
                    System.out.println();
                }
            }
            System.out.println("\nPerforming gradient descent...\n");
            Double alpha = 0.0;
            for(int i = 0; i < thetaMatrix.length; i++) {
                for(int j = 0; j < xMatrix[0].length; j++) {
                    for(int k = 0; k < yVector.length; k++) {
                    
                        System.out.println("Theta" + j + " = " + thetaMatrix[i][j]);
                        // System.out.println("x" + j + " = " + xMatrix[j][k]);
                        System.out.println("h\u019F(x) = " + h[j][k]);
                        // LEFT OFF HERE! WATCH VIDEOS FIRST!
                    }
                }
                System.out.println();
            }
            */
        }
    }
}
