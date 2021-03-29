#include "CollisionDetector.h"
#include "mesh.h"
#include <algorithm>

void CollisionDetector::addObject(TetMesh* mesh)
{
	//add object(mesh)

	assert(mesh != nullptr && mesh != NULL);
	this->m_obj_list.push_back(mesh);
	unsigned int point_num = mesh->m_positions.size();

	vector<vector<int>> adj_list(point_num);
	
	for (int i = 0; i < mesh->m_edge_list.size(); i++) {
		adj_list[mesh->m_edge_list[i].m_v1].push_back(mesh->m_edge_list[i].m_v2);
		adj_list[mesh->m_edge_list[i].m_v2].push_back(mesh->m_edge_list[i].m_v1);
	}
	topologyList.push_back(adj_list);
	PointState p{
		-1.0f,
		glm::vec3(0),
		NON_COLLIDE
	};
	vector<PointState> pointState(point_num, p);
	pointStateTable.push_back(pointState);
	//TODO: topology data 
}

void CollisionDetector::setHashParam(int p1, int p2, int p3, int tableSize, float gridSize)
{
	this->p1 = p1;
	this->p2 = p2;
	this->p3 = p3;
	this->n = tableSize;
	this->gridSize = gridSize;
}

void CollisionDetector::setGridSize(float gridSize)
{
	this->gridSize = gridSize;
}

vector<CollisionInfo> CollisionDetector::detectCollision(VectorX& x)
{


	/*
	/////////////////  Pseudo Code /////////////////////

	map all vertices into hashTable

	for each object(mesh){
		for each tetrahedron
		{
			calculate aabb of tetra;
			calculate overlapping cell;
			get candidate colliding vertices in the overlapping cell using hashTable;

			if(tetra and candidate point intersets){
				add collision info to return list;
				
			}

			 
		}
	}
		timestamp++;

	///////////////////////////////////////////////////
	*/

	for (int i = 0; i < m_obj_list.size(); i++) {

		Eigen2GLM(x, m_obj_list[i]->m_positions);

	}

	vector<CollisionInfo> ret;
	
	initPointStateTable();
	mapVertices();
	mapFaces();

	
	//Let's find collision point

	//iter all mesh
	for (int mesh_inx = 0; mesh_inx < m_obj_list.size(); mesh_inx++) {
		

		TetMesh* penetrated = m_obj_list[mesh_inx];
	
		// iter each tetrahedron
		for (int tet_inx = 0; tet_inx < penetrated->m_tet_list.size(); tet_inx++) {
			

			glm::vec3 aabb_min;
			glm::vec3 aabb_max;

			//calculate aabb of tetrahedron
			calculateTetAABB(penetrated, tet_inx, aabb_min, aabb_max);
			

			//iter overlapping cell of aabb
			for (float x = aabb_min.x; x <= aabb_max.x+gridSize; x+=gridSize) {
				for (float y = aabb_min.y; y <= aabb_max.y+gridSize;  y+= gridSize) {
					for (float z =aabb_min.z; z <= aabb_max.z+gridSize; z+= gridSize) { 

						//calculate hashkey
						int key = calculateKey(x, y, z);

						for (list<MappedVertice>::iterator iter = hashTableV[key].begin() ; iter != hashTableV[key].end() && iter->timeStamp == timeStamp; iter++) {

							// penetrating candidate vertex 
							int penetrating_index = iter->obj_index;
							int vertex_index = iter->vectice_index;
							glm::vec3 point = m_obj_list[penetrating_index]->m_positions[vertex_index];
							
							//check intersection test between penetrated tetrahedron and point
							if (IsIntersectTetPoint(penetrated,tet_inx, point )) {
								
								bool find = false;
								
								for (int col_inx = 0; col_inx < ret.size(); col_inx++) {
								

									if (ret[col_inx].penetratingMesh == m_obj_list[penetrating_index] && ret[col_inx].penetratedMesh == penetrated) {
										find = true;
										ret[col_inx].verticeList.push_back(vertex_index);
										break;
									}

								}
								if (!find) {
								
									ret.push_back(
										CollisionInfo{
											penetrated,
											m_obj_list[penetrating_index],
											mesh_inx,
											penetrating_index,
											vector<int>(1,vertex_index),
											vector<float>(),
											vector<glm::vec3>()
										}
									);
								}
								


							}


						}




					}
				}
			}



		}





	}

	
	// make all vertice vector unique
	for (auto& collisioninfo : ret) makeVectorUnique(collisioninfo.verticeList);

	
	for (auto& collisioninfo : ret) {
		collisioninfo.penetrationDepth.resize(collisioninfo.verticeList.size(), 0.0f);
		collisioninfo.penetrationDirection.resize(collisioninfo.verticeList.size(), glm::vec3(0));
	}
	/*
	
	1. identify all collision point (completed)
	2. identify all intersection edge
	3. calcuculate exact point , normal
	4. propagate
	
	*/

	computePenetration(ret);

	

	// after 1000 iter, clean outdated data
	if (timeStamp % 1000 == 999) {
		cleanHashTable();
		timeStamp = 0;
	}

	timeStamp++;
	previous = ret;
	return ret;
}

vector<CollisionInfo> CollisionDetector::detectSelfCollision(VectorX& x)
{
	return vector<CollisionInfo>();
}

void CollisionDetector::DrawCollisionPoint(const VBO& vbos)
{
	Cube cube(DEFAULT_SELECTION_RADIUS, DEFAULT_SELECTION_RADIUS, DEFAULT_SELECTION_RADIUS);
	
	cube.change_color(glm::vec3(1.0f, 0.0f, 0.0f)); // highlight using reed
	
	
	glm::vec3 m_x_i;

	for (unsigned int i = 0; i < previous.size(); i++) {

		for (unsigned int j = 0; j < previous[i].verticeList.size(); j++) {

			int pos_index = previous[i].verticeList[j];

			m_x_i = previous[i].penetratingMesh->m_positions[pos_index];
			cube.move_to(m_x_i);
			cube.Draw(vbos);


		}


	}



}

void CollisionDetector::DrawPenetration(const VBO& vbos)
{
	vector<glm::vec3> positionVector;
	vector<unsigned int> elementVector;
	int num = 1;
	glLineWidth(3.0f);
	
	for (int i = 0; i < previous.size(); i++) {
		
		for (int j = 0; j < previous[i].verticeList.size(); j++) {
			int meshIndex = i;
			int vIndex = previous[i].verticeList[j];

			float depth = previous[i].penetrationDepth[j];
			glm::vec3 dir = previous[i].penetrationDirection[j];

			glm::vec3 p1 = m_obj_list[meshIndex]->m_positions[vIndex];
			glm::vec3 p2 = p1 + depth * dir;

			positionVector.push_back(p1);
			positionVector.push_back(p2);
			
			elementVector.push_back(num++);
			elementVector.push_back(num++);

		}
	}

	unsigned int vertexSize = positionVector.size();
	unsigned int element_num = elementVector.size();

	// position
	glBindBuffer(GL_ARRAY_BUFFER, vbos.m_vbo);
	glBufferData(GL_ARRAY_BUFFER, 3*vertexSize * sizeof(float), &positionVector[0], GL_DYNAMIC_DRAW);

	// color
	std::vector<glm::vec3> colors;
	colors.resize(positionVector.size());
	std::fill(colors.begin(), colors.end(), glm::vec3(1.0f, 0, 0));
	glBindBuffer(GL_ARRAY_BUFFER, vbos.m_cbo);
	glBufferData(GL_ARRAY_BUFFER, 3*vertexSize * sizeof(float), &colors[0], GL_STATIC_DRAW);

	// indices
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos.m_ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, element_num * sizeof(unsigned int), &elementVector[0], GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, vbos.m_vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vbos.m_cbo);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glm::mat4 identity = glm::mat4(); // identity matrix
	glUniformMatrix4fv(vbos.m_uniform_transformation, 1, false, &identity[0][0]);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbos.m_ibo);
	glDrawArrays(GL_LINES,0, element_num);
	//glDrawElements(GL_TRIANGLES, element_num, GL_UNSIGNED_INT, 0);
	

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


}

void CollisionDetector::mapVertices()
{
	/*
		/// Pseodo code ///

		for all the vertices{
			map to HashTable
		}

	*/

	for (int i = 0; i < m_obj_list.size(); i++) {

		for (int v = 0; v < m_obj_list[i]->m_positions.size(); v++) {

			glm::vec3 pos = m_obj_list[i]->m_positions[v];
			int key = calculateKey(pos.x, pos.y, pos.z);

			hashTableV[key].push_front(
				MappedVertice{
					i,
					v,
					timeStamp
				}
			);



		}



	}
}

void CollisionDetector::mapFaces()
{
	static int m = 1;
	
	for (int i = 0; i < m_obj_list.size(); i++) {

		for (int f = 0; 3*f < m_obj_list[i]->m_triangle_list.size(); ++f) {
			
			glm::vec3 aabb_max, aabb_min;

			calculateTriAABB(m_obj_list[i], f, aabb_min, aabb_max);
			m++;
			if (m % 100 == 99) {
				int index1 = m_obj_list[i]->m_triangle_list[3 * f];
				int index2 = m_obj_list[i]->m_triangle_list[3 * f+1];
				int index3 = m_obj_list[i]->m_triangle_list[3 * f+2];

				glm::vec3 p1 = m_obj_list[i]->m_positions[index1];
				glm::vec3 p2 = m_obj_list[i]->m_positions[index2];

				glm::vec3 p3 = m_obj_list[i]->m_positions[index3];

	
			}


			for (float x = aabb_min.x; x <= aabb_max.x + gridSize; x += gridSize) {
				for (float y = aabb_min.y; y <= aabb_max.y + gridSize; y += gridSize) {
					for (float z = aabb_min.z; z <= aabb_max.z + gridSize; z += gridSize) {


						int key = calculateKey(x, y, z);
						
						hashTableF[key].push_front(
							MappedFace{
								i,
								f,
								timeStamp
							}
						);




					}
				}
			}

		}



	}
	
}

int CollisionDetector::calculateKey(float x, float y, float z)
{
	// calculate hashkey //

	// input: spatial position x ,y, z
	// output: Hash key

	
	int a1 = ((int)floorf(x / gridSize)) * p1;
	int a2 = ((int)floorf(y / gridSize)) * p2;
	int a3 = ((int)floorf(z / gridSize)) * p3;
	//printf_s("calculate key input:  (x: %f, y: %f, z: %f)  output: (a1: %d, a2: %d, a3: %d) \n", x, y, z, ((int)floorf(x / gridSize)), ((int)floorf(y / gridSize)), ((int)floorf(z / gridSize)));

	int ret= ( a1 ^ a2 ^ a3 ) % n;

	

	if(ret<0 ) ret+=n;
	return ret;
}

bool CollisionDetector::IsIntersectTetPoint(TetMesh* mesh, unsigned int tet_index, glm::vec3& point)
{
	/*
	* check intersect bewtween tetrahedron and vertex
	*/

	glm::vec3 x1 = mesh->m_positions[mesh->m_tet_list[tet_index].id1];
	glm::vec3 x2 = mesh->m_positions[mesh->m_tet_list[tet_index].id2];
	glm::vec3 x3 = mesh->m_positions[mesh->m_tet_list[tet_index].id3];
	glm::vec3 x4 = mesh->m_positions[mesh->m_tet_list[tet_index].id4];

	//if point is member of tetrahedron, intersection test fail.
	if (checkSamePoint(x1, point) || checkSamePoint(x2, point) || checkSamePoint(x3, point) || checkSamePoint(x4, point)) return false;


	//set barycentric coordinate (calculate barycentric basis)
	glm::vec3 barycentricV1 = x2 - x1;
	glm::vec3 barycentricV2 = x3 - x1;
	glm::vec3 barycentricV3 = x4 - x1;
	glm::vec3 p_x0= point -x1; //p-x0 vector

	glm::mat3x3 A(barycentricV1, barycentricV2, barycentricV3);

	//TODO: what if non invertibe?
	glm::mat3x3 Ainverse= glm::inverse(A);
	glm::vec3 solution = Ainverse * p_x0;

	if (solution.x > 0 && solution.y > 0 && solution.z > 0 && solution.x + solution.y + solution.z < 1) return true;
	return false;
}

bool CollisionDetector::IsIntersectTriLine(TetMesh* mesh, int tri_index, glm::vec3 pStart,glm::vec3 pEnd,glm::vec3& exactPoint)
{


	

	

	glm::vec3 p1 = mesh->m_positions[mesh->m_triangle_list[3 * tri_index]];
	glm::vec3 p2 = mesh->m_positions[mesh->m_triangle_list[3 * tri_index+1]];
	glm::vec3 p3 = mesh->m_positions[mesh->m_triangle_list[3 * tri_index+2]];


	// check if member of triangle
	if (
		(checkSamePoint(pStart, p1) || checkSamePoint(pStart, p2) || checkSamePoint(pStart, p3))
		||
		(checkSamePoint(pEnd, p1) || checkSamePoint(pEnd, p2) || checkSamePoint(pEnd, p3))
		) {
		return false;

	}
	/*if (checkSamePoint(pStart, p1) || checkSamePoint(pEnd, p1)){

		exactPoint=p1;
		return true;
	}
	if (checkSamePoint(pStart, p2) || checkSamePoint(pEnd, p2)) {

		exactPoint = p2;
		return true;
	}
	if (checkSamePoint(pStart, p3) || checkSamePoint(pEnd, p3)) {

		exactPoint = p3;
		return true;
	}*/
	
	



	glm::vec3 e = pStart - p1;
	glm::vec3 e1 = p2 - p1;
	glm::vec3 e2 = p3 - p1;
	glm::vec3 d = pEnd - pStart;
	

	float b1, b2, t;

	float coeff = 1.f / (glm::dot(glm::cross(d, e2), e1));
	
	b1 = coeff * (glm::dot(glm::cross(d, e2), e));
	b2 = coeff * (glm::dot(glm::cross(e, e1), d));
	t = coeff * (glm::dot(glm::cross(e, e1), e2));

	

	if (b1 >= 0 && b2 >= 0 && b1 + b2 <= 1 && t >= 0 && t <= 1) {
		exactPoint = pStart + t * d;
		return true;
	}
	else return false;

}

void CollisionDetector::calculateTetAABB(TetMesh* mesh, unsigned int tet_index,glm::vec3& minout,glm::vec3& maxout)
{

	/*
	*  calculate aabb of tetrahedron
	* 
	*
	*/
	float minx, miny, minz, maxx, maxy, maxz;

	
	glm::vec3 p1 = mesh->m_positions[mesh->m_tet_list[tet_index].id1];
	glm::vec3 p2 = mesh->m_positions[mesh->m_tet_list[tet_index].id2];
	glm::vec3 p3 = mesh->m_positions[mesh->m_tet_list[tet_index].id3];
	glm::vec3 p4 = mesh->m_positions[mesh->m_tet_list[tet_index].id4];

	minx = min({p1.x,p2.x,p3.x,p4.x});
	miny = min({ p1.y,p2.y,p3.y,p4.y });
	minz = min({ p1.z,p2.z,p3.z,p4.z });

	maxx = max({ p1.x,p2.x,p3.x,p4.x });
	maxy = max({ p1.y,p2.y,p3.y,p4.y });
	maxz = max({ p1.z,p2.z,p3.z,p4.z });

	minout.x = minx;
	minout.y = miny;
	minout.z = minz;

	maxout.x = maxx;
	maxout.y = maxy;
	maxout.z = maxz;


	return ;
}

void CollisionDetector::calculateTriAABB(TetMesh* mesh, unsigned int tri_index, glm::vec3& minout, glm::vec3& maxout)
{
	/*
	*  calculate aabb of triangle
	*
	*
	*/
	
	float minx, miny, minz, maxx, maxy, maxz;

	
	glm::vec3 p1 = mesh->m_positions[mesh->m_triangle_list[3 * tri_index]];
	glm::vec3 p2 = mesh->m_positions[mesh->m_triangle_list[3 * tri_index+1]];
	glm::vec3 p3 = mesh->m_positions[mesh->m_triangle_list[3 * tri_index+2]];

	/*printf_s("aabb p1: (%f,%f,%f)\n ", p1.x, p1.y, p1.z);
	printf_s("aabb p2: (%f,%f,%f)\n ", p2.x, p2.y, p2.z);
	printf_s("aabb p3: (%f,%f,%f)\n ", p3.x, p3.y, p3.z);*/

	minx = min({ p1.x,p2.x,p3.x});
	miny = min({ p1.y,p2.y,p3.y});
	minz = min({ p1.z,p2.z,p3.z});

	maxx = max({ p1.x,p2.x,p3.x});
	maxy = max({ p1.y,p2.y,p3.y});
	maxz = max({ p1.z,p2.z,p3.z});

	minout.x = minx;
	minout.y = miny;
	minout.z = minz;

	maxout.x = maxx;
	maxout.y = maxy;
	maxout.z = maxz;

	/*printf_s("aabb min: (%f,%f,%f)\n ", minout.x, minout.y, minout.z);
	printf_s("aabb max: (%f,%f,%f)\n ", maxout.x, maxout.y, maxout.z);*/
	
	
	return;
}

void CollisionDetector::voxelTraversal(glm::vec3 pStart, glm::vec3 pEnd, vector<int>& hashkeyList)
{
	
	float tDeltaX, tDeltaY, tDeltaZ;
	float tMaxX, tMaxY, tMaxZ;
	glm::vec3 d = pEnd - pStart;
	vector<float> tList;

	tDeltaX = abs( gridSize / (pEnd.x - pStart.x));
	tDeltaY = abs(gridSize / (pEnd.y - pStart.y));
	tDeltaZ = abs(gridSize / (pEnd.z - pStart.z));

	
	//TODO: hash key list return

	tMaxX = tDeltaX * (1.0 - (pStart.x / gridSize - floorf(pStart.x / gridSize)));
	tMaxY = tDeltaY * (1.0 - (pStart.y / gridSize - floorf(pStart.y / gridSize)));
	tMaxZ = tDeltaZ * (1.0 - (pStart.z / gridSize - floorf(pStart.z / gridSize)));

	//tlist
	//assert(tMaxX >= 0.0f && tMaxX < 1.0f && tMaxY >= 0.0f && tMaxY < 1.0f && tMaxZ >= 0.0f && tMaxZ < 1.0f);

	tList.push_back(0.0f);
	while (min({tMaxX,tMaxY,tMaxZ}) <1.0f ) {
		
	
		if (tMaxX < tMaxY) {
			if (tMaxX < tMaxZ) {
				tList.push_back(tMaxX);
				tMaxX += tDeltaX;

			}
			else {
				tList.push_back(tMaxZ);
				tMaxZ += tDeltaZ;

			}
		}
		else {
			if (tMaxY < tMaxZ) {
				tList.push_back(tMaxY);
				tMaxY += tDeltaY;
			}
			else {
				tList.push_back(tMaxZ);
				tMaxZ += tDeltaZ;
			}
		}


	}
	tList.push_back(1.0f);
	
	for (unsigned int i = 0; i < tList.size(); i++) {
		glm::vec3 vPoint;
		
		
		vPoint = pStart + (tList[i] + epsilone) * d;
		int key = calculateKey(vPoint.x, vPoint.y, vPoint.z);
		hashkeyList.push_back(key);
		
	}

	return;

}

void CollisionDetector::cleanHashTable()
{
	/*
	*  clean hash table
	*/


	for (int i = 0; i < n; i++) {
		hashTableV[i].clear();
		hashTableF[i].clear();
	}
	std::cout << "Hash table cleaned!\n";
	

}

bool CollisionDetector::checkSamePoint(glm::vec3& point1, glm::vec3& point2)
{
	//check if two point are the same.

	if (abs(point1.x - point2.x) < epsilone && abs(point1.y - point2.y) < epsilone && abs(point1.z - point2.z) < epsilone) return true;
	return false;
}

void CollisionDetector::makeVectorUnique(vector<int>& v)
{
	//make vector v unique (erase all the duplicated data)
	sort(v.begin(), v.end());
	v.erase(unique(v.begin(), v.end()), v.end());

}

void CollisionDetector::initPointStateTable()
{
	
	for (int i = 0; i < pointStateTable.size(); i++) {
		unsigned int num = pointStateTable[i].size();
		pointStateTable[i].resize(num, PointState{ 0.0f,glm::vec3(0),NON_COLLIDE });
		
	}
}

void CollisionDetector::computePenetration(vector<CollisionInfo>& collisionInfo)
{
	
	// mark colliding point
	for (int i = 0; i < collisionInfo.size(); i++) {
		for (int j = 0; j < collisionInfo[i].verticeList.size(); j++) {

			pointStateTable[collisionInfo[i].penetratingIndex][collisionInfo[i].verticeList[j]] = PointState{ 0.0f,glm::vec3(0),COLLIDE_NOT_PROCESSED };

		}
	}

	for (int i = 0; i < collisionInfo.size(); i++) {
		for (int j = 0; j < collisionInfo[i].verticeList.size(); j++) {

			int meshIndex = collisionInfo[i].penetratingIndex;
			int vertexIndex = collisionInfo[i].verticeList[j];

			vector<float> w;
			vector<glm::vec3> xi_p;
			vector<glm::vec3> n;
			
			for (int k = 0; k < topologyList[meshIndex][vertexIndex].size(); k++) {
				int adjIndex = topologyList[meshIndex][vertexIndex][k];
			
				//TODO: compute penetration, propagate penetration
				if (pointStateTable[meshIndex][vertexIndex].isProcessed == COLLIDE_NOT_PROCESSED && pointStateTable[meshIndex][adjIndex].isProcessed == NON_COLLIDE) {

					glm::vec3 startP = m_obj_list[meshIndex]->m_positions[vertexIndex];
					glm::vec3 endP = m_obj_list[meshIndex]->m_positions[adjIndex];
					glm::vec3 normal;
					glm::vec3 exactPoint;

					vector<int> candidateHashKeyList;
					voxelTraversal(startP, endP, candidateHashKeyList);
					makeVectorUnique(candidateHashKeyList);

					bool isFind=findNormalAndPoint(startP, endP, candidateHashKeyList, normal, exactPoint);
				
					if (isFind) {
						float w_;
						if (checkSamePoint(startP, exactPoint)) {
							
							w_ = 1.f / (glm::length(endP - exactPoint) * glm::length(endP - exactPoint));
							w.push_back(w_);
							xi_p.push_back(exactPoint - endP);
							n.push_back(normal);

							printf_s("v index: %d, adjindex: %d\n", vertexIndex, adjIndex);
						

						}
						else {
							w_ = 1.f / (glm::length(startP - exactPoint) * glm::length(startP - exactPoint));
							w.push_back(w_);
							xi_p.push_back(exactPoint - startP);
							n.push_back(normal);
					
						}

						

					}
	

				}
			}
			

			collisionInfo[i].penetrationDepth[j] = weightAverageD(w, xi_p, n);
			collisionInfo[i].penetrationDirection[j] = weightAverageN(w, n);
			
			pointStateTable[meshIndex][vertexIndex].isProcessed = COLLIDE_PROCESSED;
			pointStateTable[meshIndex][vertexIndex].penetrationDepth = collisionInfo[i].penetrationDepth[j];
			pointStateTable[meshIndex][vertexIndex].penetrationDirection = collisionInfo[i].penetrationDirection[j];
			
			printf_s("mesh index: %d, vertex index: %d collide processed\n",meshIndex,vertexIndex);
	




		}
	}

	propagatePenetration();
	
	updateCollisionInfo(collisionInfo);


}

void CollisionDetector::computePenetrationBf(vector<CollisionInfo>& collisionInfo)
{

	for (int i = 0; i < collisionInfo.size(); i++) {



		for (int j = 0; j < collisionInfo[i].verticeList.size(); j++) {

			int vertexIndex = collisionInfo[i].verticeList[j];
			int penetratingIndex = collisionInfo[i].penetratingIndex;
			int penetratedIndex = collisionInfo[i].penetratedIndex;

			glm::vec3 collidingPoint = m_obj_list[penetratingIndex]->m_positions[vertexIndex];
			glm::vec3 shortestPoint;
			

			findShortest(m_obj_list[penetratedIndex], collidingPoint, shortestPoint);
			
			float depth = glm::length(shortestPoint - collidingPoint);
			glm::vec3 direction = glm::normalize(shortestPoint - collidingPoint);

			collisionInfo[i].penetrationDepth[j] = depth;
			collisionInfo[i].penetrationDirection[j] = direction;

		}




	}




}

void CollisionDetector::propagatePenetration()
{
	
	queue<pair<int, int>> q;
	//printf_s("initializing \n");

	for (int i = 0; i < pointStateTable.size(); i++) {
		for (int j = 0; j < pointStateTable[i].size(); j++) {

			if (pointStateTable[i][j].isProcessed == COLLIDE_PROCESSED) {
				
				int meshIndex = i;
				int vertexIndex = j;

				for (int v = 0; v < topologyList[meshIndex][vertexIndex].size(); v++) {
					int adjIndex = topologyList[meshIndex][vertexIndex][v];

					if (pointStateTable[meshIndex][adjIndex].isProcessed == COLLIDE_NOT_PROCESSED) {
						q.push(pair<int, int>(meshIndex, adjIndex));
					}
				}

			}
			/*else if (pointStateTable[i][j].isProcessed == COLLIDE_NOT_PROCESSED) {
				printf_s("mesh index : %d, v index: %d collide not processed\n", i, j);
				
			}*/




		}
	}
	//("initialize ended\n");
	//if (q.empty()) {
	//	printf_s("something wrong!\n");
	//}
	while (!q.empty()) {
		
		pair<int, int> popData = q.front();
		q.pop();
		printf_s("df\n");
		int meshIndex = popData.first;
		int vIndex = popData.second;

		if (pointStateTable[meshIndex][vIndex].isProcessed == COLLIDE_NOT_PROCESSED) {
			processCollidingPoint(meshIndex, vIndex);
			pointStateTable[meshIndex][vIndex].isProcessed = COLLIDE_PROCESSED;
			for (int i = 0; i < topologyList[meshIndex][vIndex].size(); i++) {
				int adjIndex = topologyList[meshIndex][vIndex][i];
				if (pointStateTable[meshIndex][adjIndex].isProcessed == COLLIDE_NOT_PROCESSED) {
					q.push(pair<int, int>(meshIndex, adjIndex));

				}
			}


		}
		else continue;




	}


}

float CollisionDetector::weightAverageD(vector<float>& w, vector<glm::vec3> xi_p, vector<glm::vec3>& n)
{
	if (w.empty()) return epsilone;

	float sum = epsilone;
	float sum2 = 0;
	int size = w.size();
	for (int i = 0; i < size; i++) {
		
		sum += w[i];
		sum2 +=w[i]* glm::dot(xi_p[i], n[i]);
	}
	//printf_s("weight Average depth: %f\n",sum2/sum);
	return sum2/sum;
}

glm::vec3 CollisionDetector::weightAverageN(vector<float>& w, vector<glm::vec3>& n)
{
	if (n.empty()) return glm::vec3(0);
	float sum = epsilone;
	glm::vec3 sum2(0);
	int size = w.size();

	for (int i = 0; i < size; i++) {
		sum += w[i];
		sum2 += w[i] * n[i];
	}

	glm::vec3 ret = sum2 / sum;

	ret = glm::normalize(ret);
	return ret;
}

void CollisionDetector::processCollidingPoint(int meshIndex, int collidingIndex)
{

	
	vector<float> depth;
	vector<glm::vec3> dir;
	vector<glm::vec3> adjPos;
	glm::vec3 p = m_obj_list[meshIndex]->m_positions[collidingIndex];
	for (int i = 0; i < topologyList[meshIndex][collidingIndex].size(); i++) {

		int adjVertex = topologyList[meshIndex][collidingIndex][i];
		if (pointStateTable[meshIndex][adjVertex].isProcessed == COLLIDE_PROCESSED) {
			float m = pointStateTable[meshIndex][adjVertex].penetrationDepth;
			glm::vec3 r= pointStateTable[meshIndex][adjVertex].penetrationDirection;
			depth.push_back(m);
			dir.push_back(r);
			adjPos.push_back(m_obj_list[meshIndex]->m_positions[adjVertex]);
		
		}


	}
	float denominator = 0;
	float numerator1 = 0;
	glm::vec3 numerator2(0.0);
	glm::vec3 r(0);

	for (int i = 0; i < depth.size(); i++) {
		float mu = 1.f / (glm::distance(adjPos[i], p) * glm::distance(adjPos[i], p));
		denominator += mu;
		numerator1 += mu * (glm::dot(adjPos[i] - p, dir[i]) + depth[i]);
		numerator2 += mu * (dir[i]);
	}
	float penetrationDepth = numerator1 / denominator;
	glm::vec3 penetrationDir = numerator2 / denominator;
	penetrationDir = glm::normalize(penetrationDir);

	pointStateTable[meshIndex][collidingIndex].penetrationDepth = penetrationDepth;
	pointStateTable[meshIndex][collidingIndex].penetrationDirection = penetrationDir;


}

bool CollisionDetector::findNormalAndPoint(glm::vec3 startP, glm::vec3 endP, vector<int>& candidateKey, glm::vec3& normal, glm::vec3& exactP)
{
	//bool isFind = false;
	//printf_s("==================findNormalAndPoint called================\n");
	for (int keyIndex = 0; keyIndex < candidateKey.size(); keyIndex++) {

		int key = candidateKey[keyIndex];
		for (list<MappedFace>::iterator iter = hashTableF[key].begin(); iter != hashTableF[key].end() && iter->timeStamp == timeStamp; iter++) {

			int faceMeshIndex = iter->obj_index;
			int faceIndex = iter->face_index;


			if (IsIntersectTriLine(m_obj_list[faceMeshIndex], faceIndex, startP, endP, exactP)) {

				int index1 = m_obj_list[faceMeshIndex]->m_triangle_list[3 * faceIndex];
				int index2 = m_obj_list[faceMeshIndex]->m_triangle_list[3 * faceIndex + 1];
				int index3 = m_obj_list[faceMeshIndex]->m_triangle_list[3 * faceIndex + 2];
				glm::vec3 p1 = m_obj_list[faceMeshIndex]->m_positions[index1];
				glm::vec3 p2 = m_obj_list[faceMeshIndex]->m_positions[index2];
				glm::vec3 p3 = m_obj_list[faceMeshIndex]->m_positions[index3];

				/*printf_s("=========\nintersection found!!\n===========\n");
				printf_s("===============\ntri line intersection!!\n");
				printf_s("tri_1: (%f,%f,%f)\n", p1.x, p1.y, p1.z);
				printf_s("tri_2: (%f,%f,%f)\n", p2.x, p2.y, p2.z);
				printf_s("tri_3: (%f,%f,%f)\n", p3.x, p3.y, p3.z);
				printf_s("line start: (%f,%f,%f)\n",startP.x, startP.y, startP.z);
				printf_s("line end: (%f,%f,%f)\n", endP.x, endP.y, endP.z);

				printf_s("exact point: (%f,%f,%f)\n===================\n", exactP.x, exactP.y, exactP.z);*/


				normal = glm::cross(p2 - p1, p3 - p2);
				normal = glm::normalize(normal);
				

				return true;
			}




		}


	}

	//printf_s("==================findNormalAndPoint ended================\n");
	return false;
}



void CollisionDetector::updateCollisionInfo(vector<CollisionInfo>& collisionInfo)
{
	for (int i = 0; i < collisionInfo.size(); i++) {
		for (int j = 0; j < collisionInfo[i].verticeList.size(); j++) {
			int penetratingIndex = collisionInfo[i].penetratingIndex;
			int vIndex = collisionInfo[i].verticeList[j];
			if (pointStateTable[penetratingIndex][vIndex].isProcessed == COLLIDE_PROCESSED) {

				collisionInfo[i].penetrationDepth[j] = pointStateTable[penetratingIndex][vIndex].penetrationDepth;
				collisionInfo[i].penetrationDirection[j] = pointStateTable[penetratingIndex][vIndex].penetrationDirection;
			}
			else {
				//std::cout << "no info about colliding depth and direction\n";
			}


		}
	}
}

void CollisionDetector::findShortest(TetMesh* mesh, glm::vec3 collidingPoint, glm::vec3& shortest)
{
	float min = 99999;

	for (int i = 0; 3*i < mesh->m_triangle_list.size(); i++) {

		int index1 = mesh->m_triangle_list[3*i];
		int index2 = mesh->m_triangle_list[3 * i+1];
		int index3 = mesh->m_triangle_list[3 * i+2];

		

		glm::vec3 p1 = mesh->m_positions[index1];
		glm::vec3 p2 = mesh->m_positions[index2];
		glm::vec3 p3 = mesh->m_positions[index3];

		if (checkSamePoint(p1, collidingPoint) || checkSamePoint(p2, collidingPoint) || checkSamePoint(p3, collidingPoint)) {
			continue;
		}

		float len = glm::length(p1- collidingPoint);
		if (len < min) {
			shortest = p1;
			min = len;
		}
		len = glm::length(p2 - collidingPoint);
		if (len < min) {
			shortest = p2;
			min = len;
		}
		len = glm::length(p3 - collidingPoint);
		if (len < min) {
			shortest = p3;
			min = len;
		}



	}




}

int CollisionDetector::debug_checkReallynoIntersection(glm::vec3 startP, glm::vec3 endP)
{
	int ret = 0;
	glm::vec3 ep(0);
	for (int i = 0; i < m_obj_list.size(); i++) {
		
		for (int f = 0; 3*f < m_obj_list[i]->m_triangle_list.size(); f++) {
			

			if (IsIntersectTriLine(m_obj_list[i], f, startP, endP, ep)) {
				ret++;
			}

		}




	}


	return ret;
}

