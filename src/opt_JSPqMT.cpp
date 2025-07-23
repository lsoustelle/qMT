#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <iostream>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <nlopt.hpp>
namespace py = pybind11;


/////////////////////////////////////////////////////////////////////
// Simulation kernel ////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

static std::vector<double> global_data;
void set_global_data(const std::vector<double>& data) {
    global_data = data;
}

Eigen::VectorXd get_steady_state_GBM(const Eigen::Matrix<double, 6, 6>& A)
{
    Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> solver(A);
    Eigen::VectorXcd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXcd eigenvectors = solver.eigenvectors();

    int max_idx = 0;
    double max_real = eigenvalues[0].real();

    for (int i = 1; i < eigenvalues.size(); ++i) {
        if (eigenvalues[i].real() > max_real) {
            max_real = eigenvalues[i].real();
            max_idx = i;
        }
    }

    // Normalize eigenvector
    Eigen::VectorXcd Mss_c = eigenvectors.col(max_idx);
    Mss_c /= Mss_c[5]; // normalize by last component

    return Mss_c.real(); // Return real part only
}

Eigen::VectorXd get_steady_state_Graham(const Eigen::Matrix<double, 5, 5>& A)
{
    Eigen::EigenSolver<Eigen::Matrix<double, 5, 5>> solver(A);
    Eigen::VectorXcd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXcd eigenvectors = solver.eigenvectors();

    int max_idx = 0;
    double max_real = eigenvalues[0].real();

    for (int i = 1; i < eigenvalues.size(); ++i) {
        if (eigenvalues[i].real() > max_real) {
            max_real = eigenvalues[i].real();
            max_idx = i;
        }
    }

    // Normalize eigenvector
    Eigen::VectorXcd Mss_c = eigenvectors.col(max_idx);
    Mss_c /= Mss_c[4]; // normalize by last component

    return Mss_c.real(); // Return real part only
}

std::vector<double>
func_JSPqMT_GBM(const std::vector<double>& xData, double R1f, double M0b)
{
    double MTw_TR       = global_data[0];
    double MTw_Ts       = global_data[1];
    double MTw_Tm       = global_data[2];
    double MTw_ROdur    = global_data[3];
    double MTw_Tr       = MTw_TR - MTw_ROdur - MTw_Tm - MTw_Ts;
    double MTw_Df       = global_data[4];
    double VFA_TR       = global_data[5];
    double VFA_ROdur    = global_data[6];
    double VFA_Tr       = VFA_TR - VFA_ROdur;
    int N_VFA           = global_data[7];
    double R            = global_data[8];
    double R1fT2f       = global_data[9];

    double MTw_SATw1RMS     = xData[0];
    double MTw_SATWb        = xData[1];
    double MTw_ROR2sl       = xData[2];
    double MTw_ROFA         = xData[3];
    std::vector<double> VFA_ROR2sl(xData.begin()+4, xData.begin()+4+N_VFA);
    std::vector<double> VFA_ROFA(xData.begin()+4+N_VFA, xData.begin()+4+2*N_VFA);
    double B0               = xData.back();

    // build matrices & compute steady-state
    double R1b          = R1f;
    double M0f          = 1-M0b;
    double R2f          = 1/(R1fT2f/R1f);
    double MTw_WfSAT    = ( std::pow(MTw_SATw1RMS/(2*M_PI*( MTw_Df+B0)),2) + 
                            std::pow(MTw_SATw1RMS/(2*M_PI*(-MTw_Df+B0)),2))/(2*(1/R2f)); // average
    
    //// Xtilde spoil
    Eigen::Vector<double, 6> diagVals;
    diagVals << 0.0, 0.0, 1.0, 0.0, 1.0, 1.0;
    Eigen::Matrix<double, 6, 6> Xt_PHI_SPOIL = diagVals.asDiagonal();

    //// Common relaxation-exchange matrix
    Eigen::Matrix<double, 6, 6> At_REX;
    ////        Mxf    Myf      Mzf            Mxb   Mzb           C
    At_REX <<   -R2f,  0,       0,             0,    0,            0,
                0,     -R2f,    0,             0,    0,            0,
                0,     0,       -(R1f+R*M0b),  0,    R*M0f,        R1f*M0f,
                0,     0,       0,             0,    0,            0,
                0,     0,       R*M0b,         0,    -(R1b+R*M0f), R1b*M0b,
                0,     0,       0,             0,    0,            0;

    //// MTw
    Eigen::Matrix<double, 6, 6> At_MTw_SAT = At_REX;
    At_MTw_SAT = At_REX;
    At_MTw_SAT(2,2) = At_MTw_SAT(2,2) - MTw_WfSAT;
    At_MTw_SAT(4,4) = At_MTw_SAT(4,4) - MTw_SATWb;

    Eigen::Matrix<double, 6, 6> At_MTw_RO = At_REX;
    double omega = MTw_ROFA/MTw_ROdur; // RO is equivalent to a BP in R2sl formalism
    At_MTw_RO = At_REX;
    At_MTw_RO(1,2) = omega;
    At_MTw_RO(2,1) = -omega;
    At_MTw_RO(3,3) = -MTw_ROR2sl;
    At_MTw_RO(3,4) = omega;
    At_MTw_RO(4,3) = -omega;

    Eigen::Matrix<double, 6, 6> Xt_MTw_SAT  = (At_MTw_SAT*MTw_Tm).exp();
    Eigen::Matrix<double, 6, 6> Xt_MTw_RO   = (At_MTw_RO*MTw_ROdur).exp();
    Eigen::Matrix<double, 6, 6> Xt_MTw_RD   = (At_REX*MTw_Ts).exp(); // resting delay SAT pulse to RO pulse
    Eigen::Matrix<double, 6, 6> Xt_MTw_TR   = (At_REX*MTw_Tr).exp(); // resting delay RO pulse to TR
    Eigen::Matrix<double, 6, 6> Xt_MT0_RD   = (At_REX*(MTw_Tm+MTw_Ts+MTw_Tr)).exp();
    Eigen::VectorXd Mss_MT0 = get_steady_state_GBM(Xt_MT0_RD * Xt_PHI_SPOIL * Xt_MTw_RO);
    Eigen::VectorXd Mss_MTw = get_steady_state_GBM(Xt_MTw_RD * Xt_MTw_SAT * Xt_MTw_TR * Xt_PHI_SPOIL * Xt_MTw_RO);
    double Mxy_MT0 = Mss_MT0[2] * std::sin(MTw_ROFA);
    double Mxy_MTw = Mss_MTw[2] * std::sin(MTw_ROFA);

    //// VFA
    Eigen::Matrix<double, 6, 6> Xt_VFA_TR = (At_REX*VFA_Tr).exp();
    std::vector<Eigen::Matrix<double, 6, 6>> At_VFA_RO(N_VFA);
    std::vector<Eigen::Matrix<double, 6, 6>> Xt_VFA_RO(N_VFA);
    Eigen::VectorXd Mss_VFA;
    std::vector<double> Mxy_VFA(N_VFA, 0.0);
    for(int ii=0; ii<N_VFA; ii++)
    {
        omega = VFA_ROFA[ii]/VFA_ROdur; // RO is equivalent to a BP in R2sl formalism
        At_VFA_RO[ii] = At_REX;
        At_VFA_RO[ii](1,2) = omega;
        At_VFA_RO[ii](2,1) = -omega;
        At_VFA_RO[ii](3,3) = -VFA_ROR2sl[ii];
        At_VFA_RO[ii](3,4) = omega;
        At_VFA_RO[ii](4,3) = -omega;
        Xt_VFA_RO[ii] = (At_VFA_RO[ii]*VFA_ROdur).exp();
        Mss_VFA = get_steady_state_GBM(Xt_VFA_TR * Xt_PHI_SPOIL * Xt_VFA_RO[ii]);
        Mxy_VFA[ii] = Mss_VFA[2] * std::sin(VFA_ROFA[ii]);
    }

    // build normalized vector (Mxy_VFA,...,Mxy_MTw) / Mxy_MT0
    std::vector<double> Mxy_norm;
    Mxy_norm.reserve(Mxy_VFA.size() + 1);
    double eps = std::numeric_limits<double>::epsilon();
    for (double val : Mxy_VFA) {
        Mxy_norm.push_back((Mxy_MT0 > eps) ? val / Mxy_MT0 : 0.0);
    }
    Mxy_norm.push_back((Mxy_MT0 > eps) ? Mxy_MTw / Mxy_MT0 : 0.0);

    return Mxy_norm;
}

std::vector<double>
func_JSPqMT_Graham(const std::vector<double>& xData, double R1f, double M0b)
{
    double MTw_TR       = global_data[0];
    double MTw_Ts       = global_data[1];
    double MTw_Tm       = global_data[2];
    double MTw_ROdur    = global_data[3];
    double MTw_Tr       = MTw_TR - MTw_ROdur - MTw_Tm - MTw_Ts;
    double MTw_Df       = global_data[4];
    double VFA_TR       = global_data[5];
    double VFA_ROdur    = global_data[6];
    double VFA_Tr       = VFA_TR - VFA_ROdur;
    int N_VFA           = global_data[7];
    double R            = global_data[8];
    double R1fT2f       = global_data[9];

    double MTw_SATw1RMS     = xData[0];
    double MTw_SATWb        = xData[1];
    double MTw_ROWb         = xData[2];
    double MTw_ROFA         = xData[3];
    std::vector<double> VFA_ROWb(xData.begin()+4, xData.begin()+4+N_VFA);
    std::vector<double> VFA_ROFA(xData.begin()+4+N_VFA, xData.begin()+4+2*N_VFA);
    double B0               = xData.back();

    // build matrices & compute steady-state
    double R1b          = R1f;
    double M0f          = 1-M0b;
    double R2f          = 1/(R1fT2f/R1f);
    double MTw_WfSAT    = ( std::pow(MTw_SATw1RMS/(2*M_PI*( MTw_Df+B0)),2) + 
                            std::pow(MTw_SATw1RMS/(2*M_PI*(-MTw_Df+B0)),2))/(2*(1/R2f)); // average
    
    //// Xtilde spoil
    Eigen::Vector<double, 5> diagVals;
    diagVals << 0.0, 0.0, 1.0, 1.0, 1.0;
    Eigen::Matrix<double, 5, 5> Xt_PHI_SPOIL = diagVals.asDiagonal();

    //// Common relaxation-exchange matrix
    Eigen::Matrix<double, 5, 5> At_REX;
    ////        Mxf    Myf      Mzf            Mzb            C
    At_REX <<   -R2f,  0,       0,             0,             0,
                0,     -R2f,    0,             0,             0,
                0,     0,       -(R1f+R*M0b),  R*M0f,         R1f*M0f,
                0,     0,       R*M0b,          -(R1b+R*M0f), R1b*M0b,
                0,     0,       0,             0,             0;

    //// MTw
    Eigen::Matrix<double, 5, 5> At_MTw_SAT = At_REX;
    At_MTw_SAT = At_REX;
    At_MTw_SAT(2,2) = At_MTw_SAT(2,2) - MTw_WfSAT;
    At_MTw_SAT(3,3) = At_MTw_SAT(3,3) - MTw_SATWb;

    Eigen::Matrix<double, 5, 5> At_MTw_RO = At_REX;
    double omega = MTw_ROFA/MTw_ROdur; // RO is equivalent to a BP in R2sl formalism
    At_MTw_RO = At_REX;
    At_MTw_RO(1,2) = omega;
    At_MTw_RO(2,1) = -omega;
    At_MTw_RO(3,3) = At_MTw_RO(3,3) - MTw_ROWb;

    Eigen::Matrix<double, 5, 5> Xt_MTw_SAT  = (At_MTw_SAT*MTw_Tm).exp();
    Eigen::Matrix<double, 5, 5> Xt_MTw_RO   = (At_MTw_RO*MTw_ROdur).exp();
    Eigen::Matrix<double, 5, 5> Xt_MTw_RD   = (At_REX*MTw_Ts).exp(); // resting delay SAT pulse to RO pulse
    Eigen::Matrix<double, 5, 5> Xt_MTw_TR   = (At_REX*MTw_Tr).exp(); // resting delay RO pulse to TR
    Eigen::Matrix<double, 5, 5> Xt_MT0_RD   = (At_REX*(MTw_Tm+MTw_Ts+MTw_Tr)).exp();
    Eigen::VectorXd Mss_MT0 = get_steady_state_Graham(Xt_MT0_RD * Xt_PHI_SPOIL * Xt_MTw_RO);
    Eigen::VectorXd Mss_MTw = get_steady_state_Graham(Xt_MTw_RD * Xt_MTw_SAT * Xt_MTw_TR * Xt_PHI_SPOIL * Xt_MTw_RO);
    double Mxy_MT0 = Mss_MT0[2] * std::sin(MTw_ROFA);
    double Mxy_MTw = Mss_MTw[2] * std::sin(MTw_ROFA);

    //// VFA
    Eigen::Matrix<double, 5, 5> Xt_VFA_TR = (At_REX*VFA_Tr).exp();
    std::vector<Eigen::Matrix<double, 5, 5>> At_VFA_RO(N_VFA);
    std::vector<Eigen::Matrix<double, 5, 5>> Xt_VFA_RO(N_VFA);
    Eigen::VectorXd Mss_VFA;
    std::vector<double> Mxy_VFA(N_VFA, 0.0);
    for(int ii=0; ii<N_VFA; ii++)
    {
        omega = VFA_ROFA[ii]/VFA_ROdur; // RO is equivalent to a BP in R2sl formalism
        At_VFA_RO[ii] = At_REX;
        At_VFA_RO[ii](1,2) = omega;
        At_VFA_RO[ii](2,1) = -omega;
        At_VFA_RO[ii](3,3) = At_VFA_RO[ii](3,3) - VFA_ROWb[ii];
            // std::cout << "At_VFA_RO[ii]:\n" << At_VFA_RO[ii] << std::endl;
        Xt_VFA_RO[ii] = (At_VFA_RO[ii]*VFA_ROdur).exp();
        Mss_VFA = get_steady_state_Graham(Xt_VFA_TR * Xt_PHI_SPOIL * Xt_VFA_RO[ii]);
        Mxy_VFA[ii] = Mss_VFA[2] * std::sin(VFA_ROFA[ii]);
    }

    // build normalized vector (Mxy_VFA,...,Mxy_MTw) / Mxy_MT0
    std::vector<double> Mxy_norm;
    Mxy_norm.reserve(Mxy_VFA.size() + 1);
    double eps = std::numeric_limits<double>::epsilon();
    for (double val : Mxy_VFA) {
        Mxy_norm.push_back((Mxy_MT0 > eps) ? val / Mxy_MT0 : 0.0);
    }
    Mxy_norm.push_back((Mxy_MT0 > eps) ? Mxy_MTw / Mxy_MT0 : 0.0);

    return Mxy_norm;
}

/////////////////////////////////////////////////////////////////////
// Optimization - nlopt /////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

struct OptimizationData 
{
    std::vector<double> xData;
    std::vector<double> yData;
};

double objective_func_GBM(const std::vector<double> &params, std::vector<double> &grad, void *data) 
{
    double R1f = params[0];
    double M0b = params[1];

    OptimizationData* opt_data = static_cast<OptimizationData*>(data);
    auto& xData = opt_data->xData;
    auto& yData = opt_data->yData;

    std::vector<double> model_output = func_JSPqMT_GBM(xData, R1f, M0b);
    double error = 0.0;

    for (int ii=0; ii<yData.size(); ii++) {
        double diff = model_output[ii] - yData[ii];
        error += diff * diff;
    }

    return error / yData.size();
}

std::vector<double> fit_JSPqMT_nlopt_GBM(const std::vector<double> &xData,const std::vector<double> &yData) 
{
    nlopt::opt opt(nlopt::LN_BOBYQA, 2);
    opt.set_lower_bounds({0.05, 0.0});
    opt.set_upper_bounds({3.0, 0.5});
    opt.set_maxeval(400);
    opt.set_xtol_rel(1e-6);
    opt.set_ftol_rel(1e-6);

    std::vector<double> x = {1.0, 0.1};
    OptimizationData opt_data = {xData, yData};
    opt.set_min_objective(objective_func_GBM, &opt_data);

    double minf;
    try {
        opt.optimize(x, minf);
    } catch (...) {
        return {0.0, 0.0};
    }

    return x;
}

double objective_func_Graham(const std::vector<double> &params, std::vector<double> &grad, void *data) 
{
    double R1f = params[0];
    double M0b = params[1];

    OptimizationData* opt_data = static_cast<OptimizationData*>(data);
    auto& xData = opt_data->xData;
    auto& yData = opt_data->yData;

    std::vector<double> model_output = func_JSPqMT_Graham(xData, R1f, M0b);
    double error = 0.0;

    for (int ii=0; ii<yData.size(); ii++) {
        double diff = model_output[ii] - yData[ii];
        error += diff * diff;
    }

    return error / yData.size();
}

std::vector<double> fit_JSPqMT_nlopt_Graham(const std::vector<double> &xData, const std::vector<double> &yData) 
{
    nlopt::opt opt(nlopt::LN_BOBYQA, 2);
    opt.set_lower_bounds({0.05, 0.0});
    opt.set_upper_bounds({3.0, 0.5});
    opt.set_maxeval(400);
    opt.set_xtol_rel(1e-6);
    opt.set_ftol_rel(1e-6);

    std::vector<double> x = {1.0, 0.1};
    OptimizationData opt_data = {xData, yData};
    opt.set_min_objective(objective_func_Graham, &opt_data);

    double minf;
    try {
        opt.optimize(x, minf);
    } catch (...) {
        return {0.0, 0.0};
    }

    return x;
}


/////////////////////////////////////////////////////////////////////
// pybind stuff /////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(opt_JSPqMT, m)
{
    // list of exposed methods
    m.def("set_global_data",            &set_global_data,           "Store global data at Python stage");
    m.def("fit_JSPqMT_nlopt_GBM",       &fit_JSPqMT_nlopt_GBM,      "Optimize using NLopt backend for JSPqMT model w/ GBM formalism");
    m.def("fit_JSPqMT_nlopt_Graham",    &fit_JSPqMT_nlopt_Graham,   "Optimize using NLopt backend for JSPqMT model w/ Graham formalism");
}