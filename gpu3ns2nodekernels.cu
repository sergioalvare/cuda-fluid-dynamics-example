

		//////////////// PRECOMPUTATIONS & TIME STEPS & GRADIENTS

__global__
void gpu3ns2node_precomptimegrad(cfloat* flow0, cfloat* timestep, cfloat* soundspeed, cfloat* pressure, cfloat* prim, cfloat* pvisc, cfloat* pkappa, cfloat* grad, cfloat* pgrad, cfloat* lim, cfloat* res, const cfloat mach, const float rey, const cfloat* vol, const cshort* xdegree, const clong* xindex, const clong* xneigh, const cfloat* xxnormal, const cfloat* flow, const cshort irk, const clong nNode){


		//clong idx0 = 0;
		//for (clong i0 = 0; i0 < nNode; i0++) {		// Loop over nodes
			const clong tid = blockDim.x * blockIdx.x + threadIdx.x;
			const clong nthreads = blockDim.x * gridDim.x;
			for (clong i0 = tid; i0 < nNode; i0 += nthreads) {
			clong m0 = i0 * 5;
			clong n0 = m0 * 3;
			flow0[m0] = (irk < 2)? flow[m0] : flow0[m0];
			flow0[m0+1] = (irk < 2)? flow[m0+1] : flow0[m0+1];
			flow0[m0+2] = (irk < 2)? flow[m0+2] : flow0[m0+2];
			flow0[m0+3] = (irk < 2)? flow[m0+3] : flow0[m0+3];
			flow0[m0+4] = (irk < 2)? flow[m0+4] : flow0[m0+4];
			timestep[i0] = 0.0;
			cfloat vel2 = (flow[m0+1] * flow[m0+1] + flow[m0+2] * flow[m0+2] + flow[m0+3] * flow[m0+3]) / (flow[m0] * flow[m0]);
			soundspeed[i0] = sqrt(GxGM1 * (flow[m0+4] / flow[m0] - 0.5 * vel2));
			pressure[i0] = GM1 * flow[m0] * (flow[m0+4] / flow[m0] - 0.5 * vel2);
			cfloat mach2 = mach * mach;
			cfloat ssm = soundspeed[i0] * mach;
			cfloat ssm2 = ssm * ssm;
			prim[m0] = flow[m0];
			prim[m0+1] = flow[m0+1] / flow[m0];
			prim[m0+2] = flow[m0+2] / flow[m0];
			prim[m0+3] = flow[m0+3] / flow[m0];
			prim[m0+4] = GAMMA * mach2 * pressure[i0] / flow[m0];
			pvisc[i0] = 1.404 * (ssm * ssm2) / ((0.404 + ssm2) * rey);
			pkappa[i0] = pvisc[i0] / (GM1 * mach2 * PRAT);
			grad[n0] = 0.0;
			grad[n0+1] = 0.0;
			grad[n0+2] = 0.0;
			grad[n0+3] = 0.0;
			grad[n0+4] = 0.0;
			grad[n0+5] = 0.0;
			grad[n0+6] = 0.0;
			grad[n0+7] = 0.0;
			grad[n0+8] = 0.0;
			grad[n0+9] = 0.0;
			grad[n0+10] = 0.0;
			grad[n0+11] = 0.0;
			grad[n0+12] = 0.0;
			grad[n0+13] = 0.0;
			grad[n0+14] = 0.0;
			pgrad[n0] = 0.0;
			pgrad[n0+1] = 0.0;
			pgrad[n0+2] = 0.0;
			pgrad[n0+3] = 0.0;
			pgrad[n0+4] = 0.0;
			pgrad[n0+5] = 0.0;
			pgrad[n0+6] = 0.0;
			pgrad[n0+7] = 0.0;
			pgrad[n0+8] = 0.0;
			pgrad[n0+9] = 0.0;
			pgrad[n0+10] = 0.0;
			pgrad[n0+11] = 0.0;
			pgrad[n0+12] = 0.0;
			pgrad[n0+13] = 0.0;
			pgrad[n0+14] = 0.0;
			lim[m0] = 1.0;
			lim[m0+1] = 1.0;
			lim[m0+2] = 1.0;
			lim[m0+3] = 1.0;
			lim[m0+4] = 1.0;
			res[m0] = 0.0;
			res[m0+1] = 0.0;
			res[m0+2] = 0.0;
			res[m0+3] = 0.0;
			res[m0+4] = 0.0;
			for (cshort k = 0; k < xdegree[i0]; k++) {	// Loop over node neighbours extended with vertices
				//clong index = idx0 + k;
				//clong i1 = xneigh[index];
				//clong j = index * 3;

				clong idx = xindex[i0] + k;
 				clong i1 = xneigh[idx];
 				clong j = idx * 3;


				clong m1 = i1 * 5;
				cfloat normal0 = (i0 < i1)? xxnormal[j] : -xxnormal[j];
				cfloat normal1 = (i0 < i1)? xxnormal[j+1] : -xxnormal[j+1];
				cfloat normal2 = (i0 < i1)? xxnormal[j+2] : -xxnormal[j+2];
				cfloat vel21 = (flow[m1+1] * flow[m1+1] + flow[m1+2] * flow[m1+2] + flow[m1+3] * flow[m1+3]) / (flow[m1] * flow[m1]);
				cfloat soundspeed1 = sqrt(GxGM1 * (flow[m1+4] / flow[m1] - 0.5 * vel21));
				cfloat pressure1 = GM1 * flow[m1] * (flow[m1+4] / flow[m1] - 0.5 * vel21);
				cfloat area = sqrt(normal0 * normal0 + normal1 * normal1 + normal2 * normal2);	// BEGIN TAU
				cfloat avg = 0.5 * (soundspeed[i0] + soundspeed1);
				cfloat wx = flow[m0+1] / flow[m0] + flow[m1+1] / flow[m1];
				cfloat wy = flow[m0+2] / flow[m0] + flow[m1+2] / flow[m1];
				cfloat wz = flow[m0+3] / flow[m0] + flow[m1+3] / flow[m1];
				cfloat v_an = 0.5 * (wx * normal0 + wy * normal1 + wz * normal2);
				cfloat lambda_c = fabs(v_an) + area * avg;					// END TAU
				//timestep[i0] += lambda_c;			// LOCAL FEEDBACK LOOP
				atomicAdd(&timestep[i0], lambda_c);

				cfloat rho_avg =  0.5 * (flow[m0] + flow[m1]);
				cfloat velx_avg = 0.5 * (flow[m0+1] + flow[m1+1]);
				cfloat vely_avg = 0.5 * (flow[m0+2] + flow[m1+2]);
				cfloat velz_avg = 0.5 * (flow[m0+3] + flow[m1+3]);
				cfloat rhoe_avg =  0.5 * (flow[m0+4] + flow[m1+4]);
				cfloat rho_x = rho_avg * normal0;
				cfloat rho_y = rho_avg * normal1;
				cfloat rho_z = rho_avg * normal2;
				cfloat velx_x = velx_avg * normal0;
				cfloat velx_y = velx_avg * normal1;
				cfloat velx_z = velx_avg * normal2;
				cfloat vely_x = vely_avg * normal0;
				cfloat vely_y = vely_avg * normal1;
				cfloat vely_z = vely_avg * normal2;
				cfloat velz_x = velz_avg * normal0;
				cfloat velz_y = velz_avg * normal1;
				cfloat velz_z = velz_avg * normal2;
				cfloat rhoe_x = rhoe_avg * normal0;
				cfloat rhoe_y = rhoe_avg * normal1;
				cfloat rhoe_z = rhoe_avg * normal2;
				/*
				grad[n0] += rho_x;		// LOCAL FEEDBACK LOOP
				grad[n0+1] += rho_y;		// LOCAL FEEDBACK LOOP
				grad[n0+2] += rho_z;		// LOCAL FEEDBACK LOOP
				grad[n0+3] += velx_x;		// LOCAL FEEDBACK LOOP
				grad[n0+4] += velx_y;		// LOCAL FEEDBACK LOOP
				grad[n0+5] += velx_z;		// LOCAL FEEDBACK LOOP
				grad[n0+6] += vely_x;		// LOCAL FEEDBACK LOOP
				grad[n0+7] += vely_y;		// LOCAL FEEDBACK LOOP
				grad[n0+8] += vely_z;		// LOCAL FEEDBACK LOOP
				grad[n0+9] += velz_x;		// LOCAL FEEDBACK LOOP
				grad[n0+10] += velz_y;		// LOCAL FEEDBACK LOOP
				grad[n0+11] += velz_z;		// LOCAL FEEDBACK LOOP
				grad[n0+12] += rhoe_x;		// LOCAL FEEDBACK LOOP
				grad[n0+13] += rhoe_y;		// LOCAL FEEDBACK LOOP
				grad[n0+14] += rhoe_z;		// LOCAL FEEDBACK LOOP
				*/
				atomicAdd(&grad[n0], rho_x);
				atomicAdd(&grad[n0+1], rho_y);
				atomicAdd(&grad[n0+2], rho_z);
				atomicAdd(&grad[n0+3], velx_x);
				atomicAdd(&grad[n0+4], velx_y);
				atomicAdd(&grad[n0+5], velx_z);
				atomicAdd(&grad[n0+6], vely_x);
				atomicAdd(&grad[n0+7], vely_y);
				atomicAdd(&grad[n0+8], vely_z);
				atomicAdd(&grad[n0+9], velz_x);
				atomicAdd(&grad[n0+10], velz_y);
				atomicAdd(&grad[n0+11], velz_z);
				atomicAdd(&grad[n0+12], rhoe_x);
				atomicAdd(&grad[n0+13], rhoe_y);
				atomicAdd(&grad[n0+14], rhoe_z);


				cfloat prim0 = flow[m1];
				cfloat prim1 = flow[m1+1] / flow[m1];
				cfloat prim2 = flow[m1+2] / flow[m1];
				cfloat prim3 = flow[m1+3] / flow[m1];
				cfloat prim4 = GAMMA * mach2 * pressure1 / flow[m1];
				cfloat prho_avg =  0.5 * (prim[m0] + prim0);
				cfloat pvelx_avg = 0.5 * (prim[m0+1] + prim1);
				cfloat pvely_avg = 0.5 * (prim[m0+2] + prim2);
				cfloat pvelz_avg = 0.5 * (prim[m0+3] + prim3);
				cfloat prhoe_avg =  0.5 * (prim[m0+4] + prim4);
				cfloat prho_x = prho_avg * normal0;
				cfloat prho_y = prho_avg * normal1;
				cfloat prho_z = prho_avg * normal2;
				cfloat pvelx_x = pvelx_avg * normal0;
				cfloat pvelx_y = pvelx_avg * normal1;
				cfloat pvelx_z = pvelx_avg * normal2;
				cfloat pvely_x = pvely_avg * normal0;
				cfloat pvely_y = pvely_avg * normal1;
				cfloat pvely_z = pvely_avg * normal2;
				cfloat pvelz_x = pvelz_avg * normal0;
				cfloat pvelz_y = pvelz_avg * normal1;
				cfloat pvelz_z = pvelz_avg * normal2;
				cfloat prhoe_x = prhoe_avg * normal0;
				cfloat prhoe_y = prhoe_avg * normal1;
				cfloat prhoe_z = prhoe_avg * normal2;
				/*
				pgrad[n0] += prho_x;		// LOCAL FEEDBACK LOOP
				pgrad[n0+1] += prho_y;		// LOCAL FEEDBACK LOOP
				pgrad[n0+2] += prho_z;		// LOCAL FEEDBACK LOOP
				pgrad[n0+3] += pvelx_x;		// LOCAL FEEDBACK LOOP
				pgrad[n0+4] += pvelx_y;		// LOCAL FEEDBACK LOOP
				pgrad[n0+5] += pvelx_z;		// LOCAL FEEDBACK LOOP
				pgrad[n0+6] += pvely_x;		// LOCAL FEEDBACK LOOP
				pgrad[n0+7] += pvely_y;		// LOCAL FEEDBACK LOOP
				pgrad[n0+8] += pvely_z;		// LOCAL FEEDBACK LOOP
				pgrad[n0+9] += pvelz_x;		// LOCAL FEEDBACK LOOP
				pgrad[n0+10] += pvelz_y;	// LOCAL FEEDBACK LOOP
				pgrad[n0+11] += pvelz_z;	// LOCAL FEEDBACK LOOP
				pgrad[n0+12] += prhoe_x;	// LOCAL FEEDBACK LOOP
				pgrad[n0+13] += prhoe_y;	// LOCAL FEEDBACK LOOP
				pgrad[n0+14] += prhoe_z;	// LOCAL FEEDBACK LOOP
				*/
				atomicAdd(&pgrad[n0], prho_x);
				atomicAdd(&pgrad[n0+1], prho_y);
				atomicAdd(&pgrad[n0+2], prho_z);
				atomicAdd(&pgrad[n0+3], pvelx_x);
				atomicAdd(&pgrad[n0+4], pvelx_y);
				atomicAdd(&pgrad[n0+5], pvelx_z);
				atomicAdd(&pgrad[n0+6], pvely_x);
				atomicAdd(&pgrad[n0+7], pvely_y);
				atomicAdd(&pgrad[n0+8], pvely_z);
				atomicAdd(&pgrad[n0+9], pvelz_x);
				atomicAdd(&pgrad[n0+10], pvelz_y);
				atomicAdd(&pgrad[n0+11], pvelz_z);
				atomicAdd(&pgrad[n0+12], prhoe_x);
				atomicAdd(&pgrad[n0+13], prhoe_y);
				atomicAdd(&pgrad[n0+14], prhoe_z);

			}
			grad[n0] /= vol[i0];
			grad[n0+1] /= vol[i0];
			grad[n0+2] /= vol[i0];
			grad[n0+3] /= vol[i0];
			grad[n0+4] /= vol[i0];
			grad[n0+5] /= vol[i0];
			grad[n0+6] /= vol[i0];
			grad[n0+7] /= vol[i0];
			grad[n0+8] /= vol[i0];
			grad[n0+9] /= vol[i0];
			grad[n0+10] /= vol[i0];
			grad[n0+11] /= vol[i0];
			grad[n0+12] /= vol[i0];
			grad[n0+13] /= vol[i0];
			grad[n0+14] /= vol[i0];
			pgrad[n0] /= vol[i0];
			pgrad[n0+1] /= vol[i0];
			pgrad[n0+2] /= vol[i0];
			pgrad[n0+3] /= vol[i0];
			pgrad[n0+4] /= vol[i0];
			pgrad[n0+5] /= vol[i0];
			pgrad[n0+6] /= vol[i0];
			pgrad[n0+7] /= vol[i0];
			pgrad[n0+8] /= vol[i0];
			pgrad[n0+9] /= vol[i0];
			pgrad[n0+10] /= vol[i0];
			pgrad[n0+11] /= vol[i0];
			pgrad[n0+12] /= vol[i0];
			pgrad[n0+13] /= vol[i0];
			pgrad[n0+14] /= vol[i0];
			//idx0 += xdegree[i0];		// LOCAL FEEDBACK LOOP
			//atomicAdd(&idx0, xdegree[i0]);
		}

}

//-----------------------------------------------------------------------------------------------------
		//////////////// GRADIENT LIMITERS
__global__
void gpu3ns2node_limflowall_v2(cfloat* lim, const cfloat* coord, const cshort* xdegree, const clong* xindex, const clong* xneigh, const cfloat* flow, const cfloat* grad, const clong nNode) {
	const clong tid = blockDim.x * blockIdx.x + threadIdx.x;
	const clong nthreads = blockDim.x * gridDim.x;
	for (clong i0 = tid; i0 < nNode; i0 += nthreads) {

		/*
		for (clong i0 = 0; i0 < nNode; i0++) {		// Loop over nodes
		clong m0 = i0 * 5;
		fmax[m0] = flow[m0];
		fmax[m0+1] = flow[m0+1];
		fmax[m0+2] = flow[m0+2];
		fmax[m0+3] = flow[m0+3];
		fmax[m0+4] = flow[m0+4];
		fmin[m0] = flow[m0];
		fmin[m0+1] = flow[m0+1];
		fmin[m0+2] = flow[m0+2];
		fmin[m0+3] = flow[m0+3];
		fmin[m0+4] = flow[m0+4];
		*/

		clong m0 = i0 * 5;
		cfloat fmax0 = flow[m0];
		cfloat fmax1 = flow[m0+1];
		cfloat fmax2 = flow[m0+2];
		cfloat fmax3 = flow[m0+3];
		cfloat fmax4 = flow[m0+4];
		cfloat fmin0 = flow[m0];
		cfloat fmin1 = flow[m0+1];
		cfloat fmin2 = flow[m0+2];
		cfloat fmin3 = flow[m0+3];
		cfloat fmin4 = flow[m0+4];

		/*
		for (cshort k = 0; k < xdegree[i0]; k++) {	// Loop over node neighbours extended with vertices
			clong idx = xindex[i0] + k;
			clong i1 = xneigh[idx];
			clong m1 = i1 * 5;
			fmax[m0] = max(fmax[m0], flow[m1]);		// LOCAL FEEDBACK LOOP
			fmax[m0+1] = max(fmax[m0+1], flow[m1+1]);	// LOCAL FEEDBACK LOOP
			fmax[m0+2] = max(fmax[m0+2], flow[m1+2]);	// LOCAL FEEDBACK LOOP
			fmax[m0+3] = max(fmax[m0+3], flow[m1+3]);	// LOCAL FEEDBACK LOOP
			fmax[m0+4] = max(fmax[m0+4], flow[m1+4]);	// LOCAL FEEDBACK LOOP
			fmin[m0] = min(fmin[m0], flow[m1]);		// LOCAL FEEDBACK LOOP
			fmin[m0+1] = min(fmin[m0+1], flow[m1+1]);	// LOCAL FEEDBACK LOOP
			fmin[m0+2] = min(fmin[m0+2], flow[m1+2]);	// LOCAL FEEDBACK LOOP
			fmin[m0+3] = min(fmin[m0+3], flow[m1+3]);	// LOCAL FEEDBACK LOOP
			fmin[m0+4] = min(fmin[m0+4], flow[m1+4]);	// LOCAL FEEDBACK LOOP
		*/
		

		//Cogido de gpu3ns2eTpointK.cu, y se ha aniadido la x en degree index y neigh para
		//que sea como en cpu3ns2node.cpp
		for (cshort k = 0; k < xdegree[i0]; k++) {	// Loop over node neighbours extended with vertices
			clong idx = xindex[i0] + k;
			clong i1 = xneigh[idx];
			clong m1 = i1 * 5;
			fmax0 = fmaxf(fmax0, flow[m1]);			// LOCAL FEEDBACK LOOP
			fmax1 = fmaxf(fmax1, flow[m1+1]);		// LOCAL FEEDBACK LOOP
			fmax2 = fmaxf(fmax2, flow[m1+2]);		// LOCAL FEEDBACK LOOP
			fmax3 = fmaxf(fmax3, flow[m1+3]);		// LOCAL FEEDBACK LOOP
			fmax4 = fmaxf(fmax4, flow[m1+4]);		// LOCAL FEEDBACK LOOP
			fmin0 = fminf(fmin0, flow[m1]);			// LOCAL FEEDBACK LOOP
			fmin1 = fminf(fmin1, flow[m1+1]);		// LOCAL FEEDBACK LOOP
			fmin2 = fminf(fmin2, flow[m1+2]);		// LOCAL FEEDBACK LOOP
			fmin3 = fminf(fmin3, flow[m1+3]);		// LOCAL FEEDBACK LOOP
			fmin4 = fminf(fmin4, flow[m1+4]);		// LOCAL FEEDBACK LOOP

		}
		for (cshort k = 0; k < xdegree[i0]; k++) {	// Loop over node neighbours extended with vertices
		/*
			clong idx = xindex[i0] + k;
			clong i1 = xneigh[idx];
			clong n0 = i0 * 3;
			clong n1 = i1 * 3;
			clong p0 = i0 * 5;
			clong q0 = p0 * 3;
			cfloat cdiff0 = coord[n1] - coord[n0];
			cfloat cdiff1 = coord[n1+1] - coord[n0+1];
			cfloat cdiff2 = coord[n1+2] - coord[n0+2];
			cfloat proj00 = 0.5 * (cdiff0 * grad[q0] + cdiff1 * grad[q0+1] + cdiff2 * grad[q0+2]);
			cfloat proj01 = 0.5 * (cdiff0 * grad[q0+3] + cdiff1 * grad[q0+4] + cdiff2 * grad[q0+5]);
			cfloat proj02 = 0.5 * (cdiff0 * grad[q0+6] + cdiff1 * grad[q0+7] + cdiff2 * grad[q0+8]);
			cfloat proj03 = 0.5 * (cdiff0 * grad[q0+9] + cdiff1 * grad[q0+10] + cdiff2 * grad[q0+11]);
			cfloat proj04 = 0.5 * (cdiff0 * grad[q0+12] + cdiff1 * grad[q0+13] + cdiff2 * grad[q0+14]);
			cfloat phi00a = (proj00 < 0)? (fmin[p0] - flow[p0]) / proj00 : 2.0;
			cfloat phi00 = (proj00 > 0)? (fmax[p0] - flow[p0]) / proj00 : phi00a;
			cfloat phi01a = (proj01 < 0)? (fmin[p0+1] - flow[p0+1]) / proj01 : 2.0;
			cfloat phi01 = (proj01 > 0)? (fmax[p0+1] - flow[p0+1]) / proj01 : phi01a;
			cfloat phi02a = (proj02 < 0)? (fmin[p0+2] - flow[p0+2]) / proj02 : 2.0;
			cfloat phi02 = (proj02 > 0)? (fmax[p0+2] - flow[p0+2]) / proj02 : phi02a;
			cfloat phi03a = (proj03 < 0)? (fmin[p0+3] - flow[p0+3]) / proj03 : 2.0;
			cfloat phi03 = (proj03 > 0)? (fmax[p0+3] - flow[p0+3]) / proj03 : phi03a;
			cfloat phi04a = (proj04 < 0)? (fmin[p0+4] - flow[p0+4]) / proj04 : 2.0;
			cfloat phi04 = (proj04 > 0)? (fmax[p0+4] - flow[p0+4]) / proj04 : phi04a;
			cfloat phi00x = phi00 * phi00;
			cfloat phi01x = phi01 * phi01;
			cfloat phi02x = phi02 * phi02;
			cfloat phi03x = phi03 * phi03;
			cfloat phi04x = phi04 * phi04;
			cfloat lim00 = (phi00x + 2 * phi00) / (phi00x + phi00 + 2);
			cfloat lim01 = (phi01x + 2 * phi01) / (phi01x + phi01 + 2);
			cfloat lim02 = (phi02x + 2 * phi02) / (phi02x + phi02 + 2);
			cfloat lim03 = (phi03x + 2 * phi03) / (phi03x + phi03 + 2);
			cfloat lim04 = (phi04x + 2 * phi04) / (phi04x + phi04 + 2);
			lim[p0] = min(lim[p0], lim00);			// LOCAL FEEDBACK LOOP
			lim[p0+1] = min(lim[p0+1], lim01);		// LOCAL FEEDBACK LOOP
			lim[p0+2] = min(lim[p0+2], lim02);		// LOCAL FEEDBACK LOOP
			lim[p0+3] = min(lim[p0+3], lim03);		// LOCAL FEEDBACK LOOP
			lim[p0+4] = min(lim[p0+4], lim04);		// LOCAL FEEDBACK LOOP
			*/

			clong idx = xindex[i0] + k;
			clong i1 = xneigh[idx];
			clong n0 = i0 * 3;
			clong n1 = i1 * 3;
			clong p0 = m0 * 3;
			cfloat cdiff0 = coord[n1] - coord[n0];
			cfloat cdiff1 = coord[n1+1] - coord[n0+1];
			cfloat cdiff2 = coord[n1+2] - coord[n0+2];
			cfloat proj00 = 0.5 * (cdiff0 * grad[p0] + cdiff1 * grad[p0+1] + cdiff2 * grad[p0+2]);
			cfloat proj01 = 0.5 * (cdiff0 * grad[p0+3] + cdiff1 * grad[p0+4] + cdiff2 * grad[p0+5]);
			cfloat proj02 = 0.5 * (cdiff0 * grad[p0+6] + cdiff1 * grad[p0+7] + cdiff2 * grad[p0+8]);
			cfloat proj03 = 0.5 * (cdiff0 * grad[p0+9] + cdiff1 * grad[p0+10] + cdiff2 * grad[p0+11]);
			cfloat proj04 = 0.5 * (cdiff0 * grad[p0+12] + cdiff1 * grad[p0+13] + cdiff2 * grad[p0+14]);
			cfloat phi00a = (proj00 < 0)? (fmin0 - flow[m0]) / proj00 : 2.0;
			cfloat phi00 = (proj00 > 0)? (fmax0 - flow[m0]) / proj00 : phi00a;
			cfloat phi01a = (proj01 < 0)? (fmin1 - flow[m0+1]) / proj01 : 2.0;
			cfloat phi01 = (proj01 > 0)? (fmax1 - flow[m0+1]) / proj01 : phi01a;
			cfloat phi02a = (proj02 < 0)? (fmin2 - flow[m0+2]) / proj02 : 2.0;
			cfloat phi02 = (proj02 > 0)? (fmax2 - flow[m0+2]) / proj02 : phi02a;
			cfloat phi03a = (proj03 < 0)? (fmin3 - flow[m0+3]) / proj03 : 2.0;
			cfloat phi03 = (proj03 > 0)? (fmax3 - flow[m0+3]) / proj03 : phi03a;
			cfloat phi04a = (proj04 < 0)? (fmin4 - flow[m0+4]) / proj04 : 2.0;
			cfloat phi04 = (proj04 > 0)? (fmax4 - flow[m0+4]) / proj04 : phi04a;
			cfloat phi00x = phi00 * phi00;
			cfloat phi01x = phi01 * phi01;
			cfloat phi02x = phi02 * phi02;
			cfloat phi03x = phi03 * phi03;
			cfloat phi04x = phi04 * phi04;
			cfloat lim00 = (phi00x + 2 * phi00) / (phi00x + phi00 + 2);
			cfloat lim01 = (phi01x + 2 * phi01) / (phi01x + phi01 + 2);
			cfloat lim02 = (phi02x + 2 * phi02) / (phi02x + phi02 + 2);
			cfloat lim03 = (phi03x + 2 * phi03) / (phi03x + phi03 + 2);
			cfloat lim04 = (phi04x + 2 * phi04) / (phi04x + phi04 + 2);
			lim[m0] = fminf(lim[m0], lim00);		// LOCAL FEEDBACK LOOP
			lim[m0+1] = fminf(lim[m0+1], lim01);		// LOCAL FEEDBACK LOOP
			lim[m0+2] = fminf(lim[m0+2], lim02);		// LOCAL FEEDBACK LOOP
			lim[m0+3] = fminf(lim[m0+3], lim03);		// LOCAL FEEDBACK LOOP
			lim[m0+4] = fminf(lim[m0+4], lim04);		// LOCAL FEEDBACK LOOP
			}
		}
}

		

		//////////////// RESIDUALS

		// Compute upwind residuals
		// Compute viscous residuals
		// Compute residuals at the nswall, ewall and sym boundaries
		// Compute residuals at the far boundary

//-----------------------------------------------------------------------------------------------------------------

__global__ 
void gpu3ns2node_res(cfloat* res, const cfloat mach, const cfloat cosalpha, const cfloat sinalpha, const cfloat* coord, const cshort* xdegree, const clong* xindex, const clong* xneigh, const cfloat* xxnormal, const cshort* xnbc, const cfloat* flow, const cfloat* pressure, const cfloat* pvisc, const cfloat* pkappa, const cfloat* grad, const cfloat* pgrad, const cfloat* lim, const clong nNode){
		//clong idx2 = 0;
		//for (clong i0 = 0; i0 < nNode; i0++) {			// Loop over nodes
		const clong tid = blockDim.x * blockIdx.x + threadIdx.x;
		const clong nthreads = blockDim.x * gridDim.x;
		for (clong i0 = tid; i0 < nNode; i0 += nthreads) {
			for (cshort k = 0; k < xdegree[i0]; k++) {	// Loop over node neighbours extended with vertices
				//clong index = idx2 + k;
				//clong i1 = xneigh[index];
				//cshort bc = xnbc[index];
				//clong j = index * 3;
 				clong idx = xindex[i0] + k;
 				clong i1 = xneigh[idx];
 				cshort bc = xnbc[idx];
				clong j = idx * 3;


				clong n0 = i0 * 3;
				clong n1 = i1 * 3;
				clong p0 = i0 * 5;
				clong p1 = i1 * 5;
				clong q0 = p0 * 3;
				clong q1 = p1 * 3;
				cfloat normal0 = (i0 < i1)? xxnormal[j] : -xxnormal[j];
				cfloat normal1 = (i0 < i1)? xxnormal[j+1] : -xxnormal[j+1];
				cfloat normal2 = (i0 < i1)? xxnormal[j+2] : -xxnormal[j+2];
				cfloat v00 = 0.5 * (coord[n1] - coord[n0]);		// Conservative solution using gradient reconstruction
				cfloat v10 = 0.5 * (coord[n0] - coord[n1]);
				cfloat v01 = 0.5 * (coord[n1+1] - coord[n0+1]);
				cfloat v11 = 0.5 * (coord[n0+1] - coord[n1+1]);
				cfloat v02 = 0.5 * (coord[n1+2] - coord[n0+2]);
				cfloat v12 = 0.5 * (coord[n0+2] - coord[n1+2]);
				cfloat pg00 = v00 * grad[q0] + v01 * grad[q0+1] + v02 * grad[q0+2];
				cfloat pg10 = v10 * grad[q1] + v11 * grad[q1+1] + v12 * grad[q1+2];
				cfloat pg01 = v00 * grad[q0+3] + v01 * grad[q0+4] + v02 * grad[q0+5];
				cfloat pg11 = v10 * grad[q1+3] + v11 * grad[q1+4] + v12 * grad[q1+5];
				cfloat pg02 = v00 * grad[q0+6] + v01 * grad[q0+7] + v02 * grad[q0+8];
				cfloat pg12 = v10 * grad[q1+6] + v11 * grad[q1+7] + v12 * grad[q1+8];
				cfloat pg03 = v00 * grad[q0+9] + v01 * grad[q0+10] + v02 * grad[q0+11];
				cfloat pg13 = v10 * grad[q1+9] + v11 * grad[q1+10] + v12 * grad[q1+11];
				cfloat pg04 = v00 * grad[q0+12] + v01 * grad[q0+13] + v02 * grad[q0+14];
				cfloat pg14 = v10 * grad[q1+12] + v11 * grad[q1+13] + v12 * grad[q1+14];
				cfloat s00 = flow[p0] + lim[p0] * pg00;
				cfloat z10 = flow[p1] + lim[p1] * pg10;
				cfloat s01 = flow[p0+1] + lim[p0+1] * pg01;
				cfloat z11 = flow[p1+1] + lim[p1+1] * pg11;
				cfloat s02 = flow[p0+2] + lim[p0+2] * pg02;
				cfloat z12 = flow[p1+2] + lim[p1+2] * pg12;
				cfloat s03 = flow[p0+3] + lim[p0+3] * pg03;
				cfloat z13 = flow[p1+3] + lim[p1+3] * pg13;
				cfloat s04 = flow[p0+4] + lim[p0+4] * pg04;
				cfloat z14 = flow[p1+4] + lim[p1+4] * pg14;
				cfloat y14 = 0.5 + 1.0 / (GAMMA * GM1 * mach * mach);
				cfloat s10 = (bc == FAR)? 1.0 : z10;
				cfloat s11 = (bc == FAR)? cosalpha : z11;
				cfloat s12 = (bc == FAR)? 0.0 : z12;
				cfloat s13 = (bc == FAR)? sinalpha : z13;
				cfloat s14 = (bc == FAR)? y14 : z14;
				cfloat area  = sqrt(normal0 * normal0 + normal1 * normal1 + normal2 * normal2);			// BEGIN TAU
				cfloat rarea = 1.0 / area;
				cfloat nx = normal0 * rarea;
				cfloat ny = normal1 * rarea;
				cfloat nz = normal2 * rarea;
				cfloat rho_l = s00;
				cfloat vx_l = s01 / s00;
				cfloat vy_l = s02 / s00;
				cfloat vz_l = s03 / s00;
				cfloat e_l = s04 / s00;
				cfloat sqv_l = vx_l * vx_l + vy_l * vy_l + vz_l * vz_l;
				cfloat a_l = sqrt(GAMMA * GM1 * (e_l - 0.5 * sqv_l));
				cfloat p_l = (a_l * a_l * rho_l) / GAMMA;
				cfloat h_l = (e_l * rho_l + p_l) / rho_l;
				cfloat rho_r = s10;
				cfloat vx_r = s11 / s10;
				cfloat vy_r = s12 / s10;
				cfloat vz_r = s13 / s10;
				cfloat e_r = s14 / s10;
				cfloat sqv_r = vx_r * vx_r + vy_r * vy_r + vz_r * vz_r;
				cfloat a_r = sqrt(GAMMA * GM1 * (e_r - 0.5 * sqv_r));
				cfloat p_r = (a_r * a_r * rho_r) / GAMMA;
				cfloat h_r = (e_r * rho_r + p_r) / rho_r;
				cfloat rrho_l = 1.0 / rho_l;
				cfloat vn_l = vx_l * nx + vy_l * ny + vz_l * nz;
				cfloat rhovx_l = rho_l * vx_l;
				cfloat rhovy_l = rho_l * vy_l;
				cfloat rhovz_l = rho_l * vz_l;
				cfloat rhoe_l = rho_l * h_l - p_l;
				cfloat vn_r = vx_r * nx + vy_r * ny + vz_r * nz;
				cfloat rhovx_r = rho_r * vx_r;
				cfloat rhovy_r = rho_r * vy_r;
				cfloat rhovz_r = rho_r * vz_r;
				cfloat rhoe_r = rho_r * h_r - p_r;
				cfloat drho = rho_r - rho_l;
				cfloat drhovx = rhovx_r - rhovx_l;
				cfloat drhovy = rhovy_r - rhovy_l;
				cfloat drhovz = rhovz_r - rhovz_l;
				cfloat drhoe = rhoe_r - rhoe_l;
				cfloat drhovn = nx * drhovx + ny * drhovy + nz * drhovz;
				cfloat weight = sqrt(rho_r * rrho_l);
				cfloat rweight1 = 1.0 / (1 + weight);
				cfloat sweight = weight * rweight1;
				cfloat h = h_l * rweight1 + sweight * h_r;
				cfloat vx = vx_l * rweight1 + sweight * vx_r;
				cfloat vy = vy_l * rweight1 + sweight * vy_r;
				cfloat vz = vz_l * rweight1 + sweight * vz_r;
				cfloat vv = vx * vx + vy * vy + vz * vz;
				cfloat vn = nx * vx + ny * vy + nz * vz;
				cfloat aa = GM1 * h - 0.5 * GM1 * vv;
				cfloat a  = sqrt(aa);
				cfloat vdrhov = vx * drhovx + vy * drhovy + vz * drhovz;
				cfloat tmp = drho * h - drhoe + vdrhov - drho * vv;
				cfloat a2 = tmp * GM1 / aa;
				cfloat a3 = (drhovn + a * drho - vn * drho - a * a2) / (2 * a);
				cfloat a1 = drho - a2 - a3;
				cfloat tmp0 = rho_l * vn_l;
				cfloat r0a = tmp0;
				cfloat r1a = tmp0 * vx_l + nx * p_l;
				cfloat r2a = tmp0 * vy_l + ny * p_l;
				cfloat r3a = tmp0 * vz_l + nz * p_l;
				cfloat r4a = tmp0 * h_l;
				cfloat speed0 = vn_l - a_l;			// First wave, vn - a
				cfloat wave1_speed = vn - a;
				cfloat wave0a = a1;
				cfloat wave1a = a1 * vx - a1 * nx * a;
				cfloat wave2a = a1 * vy - a1 * ny * a;
				cfloat wave3a = a1 * vz - a1 * nz * a;
				cfloat wave4a = a1 * h - a1 * vn * a;
				cfloat wrho = rho_l + wave0a;
				cfloat wrhovx = rhovx_l + wave1a;
				cfloat wrhovy = rhovy_l + wave2a;
				cfloat wrhovz = rhovz_l + wave3a;
				cfloat wrhoe = rhoe_l + wave4a;
				cfloat rwrho  = 1.0 / wrho;
				cfloat wrhovn = nx * wrhovx + ny * wrhovy + nz * wrhovz;
				cfloat tmp1 = (wrhovx * wrhovx) + (wrhovy * wrhovy) + (wrhovz * wrhovz);
				cfloat wp = GM1 * wrhoe - GM1 * 0.5 * tmp1 * rwrho;
				cfloat wc = sqrt(GAMMA * wp * rwrho);
				cfloat ws = wrhovn * rwrho - wc;
				cfloat tmp2 = (wave1_speed < 0.0)? wave1_speed : 0.0;
				cfloat tmp3 = (speed0 * ws - speed0 * wave1_speed) / (ws - speed0);
				cfloat tmp4 = (speed0 < 0)? tmp3 : tmp2;
				cfloat tmp5 = (ws > 0)? tmp4 : tmp2;
				cfloat r0b = r0a + tmp5 * wave0a;
				cfloat r1b = r1a + tmp5 * wave1a;
				cfloat r2b = r2a + tmp5 * wave2a;
				cfloat r3b = r3a + tmp5 * wave3a;
				cfloat r4b = r4a + tmp5 * wave4a;
				cfloat activate = (vn > 0)? 0 : 1;		// Second wave, vn
				cfloat wave2_speed = activate * vn;
				cfloat tmp6 = drhovn - vn * drho;
				cfloat wave0b = a2;
				cfloat wave1b = a2 * vx - drho * vx + drhovx - nx * tmp6;
				cfloat wave2b = a2 * vy - drho * vy + drhovy - ny * tmp6;
				cfloat wave3b = a2 * vz - drho * vz + drhovz - nz * tmp6;
				cfloat wave4b = 0.5 * a2 * vv - drho * (vv - vn * vn) + vdrhov - vn * drhovn;
				cfloat r0c = r0b + wave2_speed * wave0b;
				cfloat r1c = r1b + wave2_speed * wave1b;
				cfloat r2c = r2b + wave2_speed * wave2b;
				cfloat r3c = r3b + wave2_speed * wave3b;
				cfloat r4c = r4b + wave2_speed * wave4b;
				cfloat wave3_speed = vn + a;			// Third wave, vn + a
				cfloat speed1 = vn_r + a_r;
				cfloat a3a = a3 * activate;
				cfloat wave0c = a3a;
				cfloat wave1c = a3a * vx + a3a * nx * a;
				cfloat wave2c = a3a * vy + a3a * ny * a;
				cfloat wave3c = a3a * vz + a3a * nz * a;
				cfloat wave4c = a3a * h + a3a * vn * a;
				cfloat wrho0 = rho_r - wave0c;
				cfloat wrhovx0 = rhovx_r - wave1c;
				cfloat wrhovy0 = rhovy_r - wave2c;
				cfloat wrhovz0 = rhovz_r - wave3c;
				cfloat wrhoe0 = rhoe_r - wave4c;
				cfloat rwrho0 = 1.0 / wrho0;
				cfloat wrhovn0 = nx * wrhovx0 + ny * wrhovy0 + nz * wrhovz0;
				cfloat tmp7 = (wrhovx0 * wrhovx0) + (wrhovy0 * wrhovy0) + (wrhovz0 * wrhovz0);
				cfloat wp0 = GM1 * wrhoe0 - 0.5 * GM1 * tmp7 * rwrho0;
				cfloat wc0 = sqrt(GAMMA * wp0 * rwrho0);
				cfloat ws0 = wrhovn0 * rwrho0 + wc0;
				cfloat tmp8 = (wave3_speed < 0.0)? wave3_speed : 0.0;
				cfloat tmp9 = (ws0 * speed1 - ws0 * wave3_speed) / (speed1 - ws0);
				cfloat tmp10 = (speed1 > 0)? tmp9 : tmp8;
				cfloat tmp11 = (ws0 < 0)? tmp10 : tmp8;
				cfloat tmp12 = tmp11 * activate * area;
				cfloat r0i = area * r0c + tmp12 * wave0c;
				cfloat r1i = area * r1c + tmp12 * wave1c;
				cfloat r2i = area * r2c + tmp12 * wave2c;
				cfloat r3i = area * r3c + tmp12 * wave3c;
				cfloat r4i = area * r4c + tmp12 * wave4c;							// END TAU
				cfloat vx0 = flow[p0+1] / flow[p0];								// BEGIN TAU
				cfloat vy0 = flow[p0+2] / flow[p0];
				cfloat vz0 = flow[p0+3] / flow[p0];
				cfloat vx1 = flow[p1+1] / flow[p1];
				cfloat vy1 = flow[p1+2] / flow[p1];
				cfloat vz1 = flow[p1+3] / flow[p1];
				cfloat vvx = 0.5 * (vx0 + vx1);
				cfloat vvy = 0.5 * (vy0 + vy1);
				cfloat vvz = 0.5 * (vz0 + vz1);
				cfloat mue_eff = 0.5 * (pvisc[i0] + pvisc[i1]);
				cfloat kappa = 0.5 * (pkappa[i0] + pkappa[i1]);
				cfloat dvx_dx = 0.5 * (pgrad[q0+3] + pgrad[q1+3]);
				cfloat dvx_dy = 0.5 * (pgrad[q0+4] + pgrad[q1+4]);
				cfloat dvx_dz = 0.5 * (pgrad[q0+5] + pgrad[q1+5]);
				cfloat dvy_dx = 0.5 * (pgrad[q0+6] + pgrad[q1+6]);
				cfloat dvy_dy = 0.5 * (pgrad[q0+7] + pgrad[q1+7]);
				cfloat dvy_dz = 0.5 * (pgrad[q0+8] + pgrad[q1+8]);
				cfloat dvz_dx = 0.5 * (pgrad[q0+9] + pgrad[q1+9]);
				cfloat dvz_dy = 0.5 * (pgrad[q0+10] + pgrad[q1+10]);
				cfloat dvz_dz = 0.5 * (pgrad[q0+11] + pgrad[q1+11]);
				cfloat dt_dx = 0.5 * (pgrad[q0+12] + pgrad[q1+12]);
				cfloat dt_dy = 0.5 * (pgrad[q0+13] + pgrad[q1+13]);
				cfloat dt_dz = 0.5 * (pgrad[q0+14] + pgrad[q1+14]);
				cfloat lambda = (-2.0/3.0) * mue_eff;
				cfloat tau_xx = lambda * (dvy_dy + dvz_dz - 2.0 * dvx_dx);
				cfloat tau_yy = lambda * (dvx_dx + dvz_dz - 2.0 * dvy_dy);
				cfloat tau_zz = lambda * (dvx_dx + dvy_dy - 2.0 * dvz_dz);
				cfloat tau_xy = mue_eff * (dvx_dy + dvy_dx);
				cfloat tau_xz = mue_eff * (dvx_dz + dvz_dx);
				cfloat tau_yz = mue_eff * (dvy_dz + dvz_dy);
				cfloat r0 = 0.0;
				cfloat r1 = -(tau_xx * normal0 + tau_xy * normal1 + tau_xz * normal2);
				cfloat r2 = -(tau_xy * normal0 + tau_yy * normal1 + tau_yz * normal2);
				cfloat r3 = -(tau_xz * normal0 + tau_yz * normal1 + tau_zz * normal2);
				cfloat r4 = -((vvx * tau_xx + vvy * tau_xy + vvz * tau_xz + kappa * dt_dx) * normal0 +
						(vvx * tau_xy + vvy * tau_yy + vvz * tau_yz + kappa * dt_dy) * normal1 +
						(vvx * tau_xz + vvy * tau_yz + vvz * tau_zz + kappa * dt_dz) * normal2);	// END TAU
				cfloat r0v = (bc != INN)? 0.0 : r0;
				cfloat r1v = (bc != INN)? 0.0 : r1;
				cfloat r2v = (bc != INN)? 0.0 : r2;
				cfloat r3v = (bc != INN)? 0.0 : r3;
				cfloat r4v = (bc != INN)? 0.0 : r4;
				cfloat r1s = -res[p0+1];
				cfloat r2s = -res[p0+2];
				cfloat r3s = -res[p0+3];
				cfloat r0t = (bc == NSWALL)? 0.0 : r0i;
				cfloat r1t = (bc == NSWALL)? r1s : r1i;
				cfloat r2t = (bc == NSWALL)? r2s : r2i;
				cfloat r3t = (bc == NSWALL)? r3s : r3i;
				cfloat r4t = (bc == NSWALL)? 0.0 : r4i;
				cfloat resid1 = pressure[i0] * normal0;
				cfloat resid2 = pressure[i0] * normal1;
				cfloat resid3 = pressure[i0] * normal2;
				bool test = (bc == EWALL) || (bc == SYM);
				cfloat r0w = (test)? 0.0 : r0t;
				cfloat r1w = (test)? resid1 : r1t;
				cfloat r2w = (test)? resid2 : r2t;
				cfloat r3w = (test)? resid3 : r3t;
				cfloat r4w = (test)? 0.0 : r4t;
				cfloat res0 = r0w + r0v;
				cfloat res1 = r1w + r1v;
				cfloat res2 = r2w + r2v;
				cfloat res3 = r3w + r3v;
				cfloat res4 = r4w + r4v;
				/*
				res[p0] += res0;		// LOCAL FEEDBACK LOOP
				res[p0+1] += res1;		// LOCAL FEEDBACK LOOP
				res[p0+2] += res2;		// LOCAL FEEDBACK LOOP
				res[p0+3] += res3;		// LOCAL FEEDBACK LOOP
				res[p0+4] += res4;		// LOCAL FEEDBACK LOOP
				*/

				atomicAdd(&res[p0], res0);
				atomicAdd(&res[p0+1], res1);
				atomicAdd(&res[p0+2], res2);
				atomicAdd(&res[p0+3], res3);
				atomicAdd(&res[p0+4], res4);
			}
			//idx2 += xdegree[i0];		// LOCAL FEEDBACK LOOP
			//atomicAdd(&idx2, xdegree[i0]);
		}
}

		//////////////// TIME INTEGRATION & ITERATION CONTROL

		// Update flow solutions
		// Compute max residuals

//-----------------------------------------------------------------------------------
__global__ 
void gpu3ns2node_maxresinit(const cfloat* res){
		/*
		*maxrho = fabs(res[0]);
		*maxrhoe = fabs(res[4]);
		*/
		gmaxrho = fabs(res[0]);
		gmaxrhoe = fabs(res[4]); //ATENCION, ESTO ES DISTINTO RESPECTO 
//AL CODIGO DE CPU. ESCRITO ASI AL COMPARAR cpu3ns2flat.cpp y gpu3ns2flatkernets.cu
//EN CONSECUENCIA, LOS SIGUIENTES KERNELS DEBEN ESTAR ADAPTADOS A ESTE CAMBIO.

}

//-----------------------------------------------------------------------------------------------------
__global__ 
//void gpu3ns2node_updatemaxres(cfloat* flow, const bool rk, const cfloat cfl, const cshort alpha, const cshort* xdegree, const cshort* xnbc, const cfloat* flow0, const cfloat* timestep, const cfloat* res, const clong nNode){
void gpu3ns2node_updatemaxres(cfloat* flow, const bool rk, const cfloat cfl, const cfloat alpha, const cshort* xdegree, const clong* xindex, const cshort* xnbc, const cfloat* flow0, const cfloat* timestep, const cfloat* res, const clong nNode){
		//clong idx3 = 0;
		//for (clong i = 0; i < nNode; i++) {		// Loop over nodes
			const clong tid = blockDim.x * blockIdx.x + threadIdx.x;
			const clong nthreads = blockDim.x * gridDim.x;
			for (clong i = tid; i < nNode; i += nthreads) {
			clong j = i * 5;
			for (cshort k = 0; k < xdegree[i]; k++) {	// Loop over node neighbours extended with vertices
				//clong index = idx3 + k;
				//cshort bc = xnbc[index];

 				clong idx = xindex[i] + k;
				cshort bc = xnbc[idx];

				flow[j+1] = (bc == NSWALL)? 0.0 : flow[j+1];	// FLOW CORRECTION
				flow[j+2] = (bc == NSWALL)? 0.0 : flow[j+2];	// FLOW CORRECTION
				flow[j+3] = (bc == NSWALL)? 0.0 : flow[j+3];	// FLOW CORRECTION
			}
			cfloat dt = cfl / timestep[i];
			cfloat srho = (rk)? flow0[j] : flow[j];
			cfloat svel0 = (rk)? flow0[j+1] : flow[j+1];
			cfloat svel1 = (rk)? flow0[j+2] : flow[j+2];
			cfloat svel2 = (rk)? flow0[j+3] : flow[j+3];
			cfloat srhoe = (rk)? flow0[j+4] : flow[j+4];
/*
			flow[j] = srho - RKALPHA[irk] * res[j] * dt;			// FLOW UPDATE
			flow[j+1] = svel0 - RKALPHA[irk] * res[j+1] * dt;		// FLOW UPDATE
			flow[j+2] = svel1 - RKALPHA[irk] * res[j+2] * dt;		// FLOW UPDATE
			flow[j+3] = svel2 - RKALPHA[irk] * res[j+3] * dt;		// FLOW UPDATE
			flow[j+4] = srhoe - RKALPHA[irk] * res[j+4] * dt;		// FLOW UPDATE
*/
			flow[j] = srho - alpha * res[j] * dt;			// FLOW UPDATE
			flow[j+1] = svel0 - alpha * res[j+1] * dt;		// FLOW UPDATE
			flow[j+2] = svel1 - alpha * res[j+2] * dt;		// FLOW UPDATE
			flow[j+3] = svel2 - alpha * res[j+3] * dt;		// FLOW UPDATE
			flow[j+4] = srhoe - alpha * res[j+4] * dt;		// FLOW UPDATE

			cfloat mrho = fabs(res[j]);
			cfloat mrhoe = fabs(res[j+4]);
			/*
			*maxrho = max(*maxrho, mrho);		// GLOBAL FEEDBACK LOOP
			*maxrhoe = max(*maxrhoe, mrhoe);	// GLOBAL FEEDBACK LOOP
			*/

			atomicMax((cint*) &gmaxrho, __float_as_int(mrho));		// GLOBAL FEEDBACK LOOP
			atomicMax((cint*) &gmaxrhoe, __float_as_int(mrhoe));		// GLOBAL FEEDBACK LOOP

			//idx3 += xdegree[i];			// LOCAL FEEDBACK LOOP
			//atomicAdd(&idx3, xdegree[i]);
		}
}


	//////////////// MONITORING

	// Compute inviscid forces
	// Compute viscous forces

//------------------------------------------------------------------------------------------------------------------------
__global__ 
void gpu3ns2node_forceinit(){
	/*
	cfloat iforce0 = 0.0;
	cfloat iforce2 = 0.0;
	cfloat vforce0 = 0.0;
	cfloat vforce2 = 0.0;
	*/
	giforce0 = 0.0;
	giforce2 = 0.0;
	gvforce0 = 0.0;
	gvforce2 = 0.0;
}

//--------------------------------------------------------------------------------------------------------------------------

__global__ 
void gpu3ns2node_force(const cshort* xdegree, const clong* xindex, const cfloat* xxnormal, const cshort* xnbc, const cfloat* pressure, const cfloat* pvisc, const cfloat* pgrad, const clong nNode){
//	clong idx4 = 0;
	//for (clong i = 0; i < nNode; i++) {		// Loop over nodes
	const clong tid = blockDim.x * blockIdx.x + threadIdx.x;
	const clong nthreads = blockDim.x * gridDim.x;
	for (clong i = tid; i < nNode; i += nthreads) {
		clong j = i * 5;
		for (cshort k = 0; k < xdegree[i]; k++) {	// Loop over node neighbours extended with vertices
			//clong index = idx4 + k;
			//cshort bc = xnbc[index];
			//clong m = index * 3;

 			clong idx = xindex[i] + k;
 			cshort bc = xnbc[idx];
			clong m = idx * 3;


			clong p = j * 3;
			bool wall = (bc == NSWALL) || (bc == EWALL);
//			cfloat if0 = giforce0 - 2.0 * pressure[i] * xxnormal[m];
//			cfloat if2 = giforce2 - 2.0 * pressure[i] * xxnormal[m+2];
			cfloat if0 = -2.0 * pressure[i] * xxnormal[m];
			cfloat if2 = -2.0 * pressure[i] * xxnormal[m+2];
			cfloat ifrc0 = (wall)? if0 : 0.0;
			cfloat ifrc2 = (wall)? if2 : 0.0;
			cfloat dvel = pgrad[p+3] + pgrad[p+7] + pgrad[p+11];
			cfloat tau00 = pvisc[i] * ((pgrad[p+3] + pgrad[p+3]) - (2.0 / 3.0) * dvel);
			cfloat tau01 = pvisc[i] * (pgrad[p+6] + pgrad[p+4]);
			cfloat tau02 = pvisc[i] * (pgrad[p+9] + pgrad[p+5]);
			cfloat tau20 = pvisc[i] * (pgrad[p+5] + pgrad[p+9]);
			cfloat tau21 = pvisc[i] * (pgrad[p+8] + pgrad[p+10]);
			cfloat tau22 = pvisc[i] * ((pgrad[p+11] + pgrad[p+11]) - (2.0 / 3.0) * dvel);
//			cfloat vf0 = gvforce0 + 2.0 * (tau00 * xxnormal[m] + tau01 * xxnormal[m+1] + tau02 * xxnormal[m+2]);
//			cfloat vf2 = gvforce2 + 2.0 * (tau20 * xxnormal[m] + tau21 * xxnormal[m+1] + tau22 * xxnormal[m+2]);
			cfloat vf0 = 2.0 * (tau00 * xxnormal[m] + tau01 * xxnormal[m+1] + tau02 * xxnormal[m+2]);
			cfloat vf2 = 2.0 * (tau20 * xxnormal[m] + tau21 * xxnormal[m+1] + tau22 * xxnormal[m+2]);
			cfloat vfrc0 = (bc == NSWALL)? vf0 : 0.0;
			cfloat vfrc2 = (bc == NSWALL)? vf2 : 0.0;
//			giforce0 = (wall)? if0 : giforce0;		// GLOBAL FEEDBACK LOOP
//			giforce2 = (wall)? if2 : giforce2;		// GLOBAL FEEDBACK LOOP
			atomicAdd(&giforce0, ifrc0);			// GLOBAL FEEDBACK LOOP
			atomicAdd(&giforce2, ifrc2);			// GLOBAL FEEDBACK LOOP
//			gvforce0 = (bc == NSWALL)? vf0 : gvforce0;	// GLOBAL FEEDBACK LOOP
//			gvforce2 = (bc == NSWALL)? vf2 : gvforce2;	// GLOBAL FEEDBACK LOOP
			atomicAdd(&gvforce0, vfrc0);			// GLOBAL FEEDBACK LOOP
			atomicAdd(&gvforce2, vfrc2);			// GLOBAL FEEDBACK LOOP
		}
//		idx4 += xdegree[i];		// LOCAL FEEDBACK LOOP
	}
}


