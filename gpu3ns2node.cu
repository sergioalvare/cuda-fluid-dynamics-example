#include "gpusolvers.hpp"

bool gpu3ns2node (cfloat* flow, cfloat* res, cfloat* maxrho, cfloat* maxrhoe, cfloat* cdrag, cfloat* clift, cint* iters, const cint maxiters, const cfloat delta, const cint refresh, const bool vk, const bool rk, const cfloat cfl, const cfloat mach, const cfloat rey, const cfloat sinalpha, const cfloat cosalpha, const clong nNode, const clong nEdge, const clong nFar, const clong nNSwall, const clong nEwall, const clong nVertex, const cfloat* coord, const cfloat* vol, const cshort* xdegree, const clong* xindex, const clong* xneigh, const cfloat* xxnormal, const cshort* xnbc) {

	//////////////// LOCAL DATA STRUCTURES

	const cfloat RKALPHA[] = {0.69, 0.1918, 0.4929, 1.0};		// Runge-Kutta coefficients (rk = true)
	cfloat* flow0;
	cfloat* timestep;
	cfloat* soundspeed;
	cfloat* pressure;
	cfloat* prim;
	cfloat* pvisc;
	cfloat* pkappa;
	cfloat* grad;
	cfloat* pgrad;
	//sint* iflow;
	//sint* iflowmax;
	//sint* iflowmin;
	cfloat* lim;
	cudaMalloc((void **) &flow0, sizeof(cfloat) * nNode * 5);	// Previous flow (rk = true)
	cudaMalloc((void **) &timestep, sizeof(cfloat) * nNode);	// Time step
	cudaMalloc((void **) &soundspeed, sizeof(cfloat) * nNode);	// Sound speed
	cudaMalloc((void **) &pressure, sizeof(cfloat) * nNode);	// Pressure
	cudaMalloc((void **) &prim, sizeof(cfloat) * nNode * 5);	// Primitive flow
	cudaMalloc((void **) &pvisc, sizeof(cfloat) * nNode);		// Primitive viscosity
	cudaMalloc((void **) &pkappa, sizeof(cfloat) * nNode);		// Primitive kappa
	cudaMalloc((void **) &grad, sizeof(cfloat) * nNode * 5 * 3);	// Gradient
	cudaMalloc((void **) &pgrad, sizeof(cfloat) * nNode * 5 * 3);	// Primitive gradient
	//cudaMalloc((void **) &iflow, sizeof(sint) * nNode * 5);		// Flow as ordered int (vk = true)
	//cudaMalloc((void **) &iflowmax, sizeof(sint) * nNode * 5);	// Max flow as ordered int (vk = true)
	//cudaMalloc((void **) &iflowmin, sizeof(sint) * nNode * 5);	// Min flow as ordered int (vk = true)
	cudaMalloc((void **) &lim, sizeof(cfloat) * nNode * 5);		// Gradient limiter (vk = true)


	//////////////// ITERATION CONTROL

	bool done = false;

	cshort nrk = (rk)? 3 : 1;
	cint nref = nrk * refresh;
	cshort rrk = 4 - nrk;
	cshort irk = rrk;

	for (cint iref = 0; iref < nref; iref++) {	// Main loop

		//////////////// PRECOMPUTATIONS
		//gpu3ns2node_precomptimegrad<<<NBLOCKS, NTHREADS>>>(flow0, timestep, soundspeed, pressure, prim, pvisc, pkappa, grad, pgrad, lim, res, mach, rey, xdegree, xindex, xneigh, xxnormal, flow, irk, nNode,vol);
gpu3ns2node_precomptimegrad<<<NBLOCKS, NTHREADS>>>(flow0, timestep, soundspeed, pressure, prim, pvisc, pkappa, grad, pgrad, lim, res, mach, rey, vol, xdegree, xindex, xneigh, xxnormal, flow, irk, nNode);


		if (vk) {
			//gpu3ns2node_limflowall<<<NBLOCKS, NTHREADS>>>(iflow, iflowmax, iflowmin, flow, nNode, xdegree, xindex, xneigh, coord, grad, lim);
			gpu3ns2node_limflowall_v2<<<NBLOCKS, NTHREADS>>>(lim, coord, xdegree, xindex, xneigh, flow, grad, nNode);

		}

		//////////////// RESIDUALS
		gpu3ns2node_res<<<NBLOCKS, NTHREADS>>>(res, mach, cosalpha, sinalpha, coord, xdegree, xindex, xneigh, xxnormal, xnbc, flow, pressure, pvisc, pkappa, grad, pgrad, lim, nNode);



		//////////////// TIME INTEGRATION & ITERATION CONTROL
		gpu3ns2node_maxresinit<<<1, 1>>>(res);
		gpu3ns2node_updatemaxres<<<NBLOCKS, NTHREADS>>>(flow, rk, cfl, RKALPHA[irk], xdegree, xindex, xnbc, flow0, timestep, res, nNode);

		cudaMemcpyFromSymbol(maxrho, gmaxrho, sizeof(cfloat), 0, cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(maxrhoe, gmaxrhoe, sizeof(cfloat), 0, cudaMemcpyDeviceToHost);


		irk = irk + 1;
		cint newiters = *iters + 1;
		*iters = (irk > 3)? newiters : *iters;
		irk = (irk > 3)? rrk : irk;

		bool stop = (*iters == maxiters)? true : false;
		done = (*maxrho < delta)? true : stop;
		if (done) break;

	}




	//////////////// MONITORING
	gpu3ns2node_forceinit<<<1, 1>>>();
	gpu3ns2node_force<<<NBLOCKS, NTHREADS>>>(xdegree, xindex, xxnormal, xnbc, pressure, pvisc, pgrad, nNode);

	cfloat iforce0;
	cfloat iforce2;
	cfloat vforce0;
	cfloat vforce2;
	cudaMemcpyFromSymbol(&iforce0, giforce0, sizeof(cfloat), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&iforce2, giforce2, sizeof(cfloat), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&vforce0, gvforce0, sizeof(cfloat), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&vforce2, gvforce2, sizeof(cfloat), 0, cudaMemcpyDeviceToHost);

	cfloat icdrag = iforce0 * cosalpha + iforce2 * sinalpha;
	cfloat iclift = -iforce0 * sinalpha + iforce2 * cosalpha;
	cfloat vcdrag = vforce0 * cosalpha + vforce2 * sinalpha;
	cfloat vclift = -vforce0 * sinalpha + vforce2 * cosalpha;
	*cdrag = icdrag + vcdrag;
	*clift = iclift + vclift;

	//////////////// LOCAL DATA STRUCTURES

	cudaFree(flow0);
	cudaFree(timestep);
	cudaFree(soundspeed);
	cudaFree(pressure);
	cudaFree(prim);
	cudaFree(pvisc);
	cudaFree(pkappa);
	cudaFree(grad);
	cudaFree(pgrad);
	//cudaFree(iflow);
	//cudaFree(iflowmax);
	//cudaFree(iflowmin);
	cudaFree(lim);

	return done;
}


