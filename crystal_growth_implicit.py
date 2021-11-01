import taichi as ti

ti.init(arch=ti.cpu)

n = 256
scatter = 2
res = n*scatter

dx, dy, dt = 0.03, 0.03, 1e-4

pi = 3.14159
tau = 3e-4
eps_bar = 0.01
sigma = 0.02
J = 6.
theta_0 = pi/2
alpha = 0.9
gamma = 10.
T_eq = 1.
kappa = 2.0
a = 0.01
coeff1 = -eps_bar * sigma * J

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res,res))

#t_max = 300
#t_min = 0

n2 = n ** 2

b = ti.field(ti.f32, shape = n2)  # we will solve a equation like Ax=b later

# p stands for phase and t stands for temperature
p_old = ti.field(ti.f32, shape = n2) 
p_new = ti.field(ti.f32, shape = n2)
t_old = ti.field(ti.f32, shape = n2) 
t_new = ti.field(ti.f32, shape = n2) 

ind = lambda i,j : n*i+j

@ti.kernel 
def init():
    for i,j in ti.ndrange(n, n):
        r = (float(i)-n//2)**2 + (float(j)-n//2)**2
        if r <= 64:
            p_old[ind(i, j)] = 1.0 
            p_new[ind(i, j)] = 1.0
            t_old[ind(i, j)] = 0.0
            t_new[ind(i, j)] = 0.0
        else:
            p_old[ind(i, j)] = 0.0
            p_new[ind(i, j)] = 0.0
            t_old[ind(i, j)] = 0.0
            t_new[ind(i, j)] = 0.0
@ti.func
def get_color(p):
    c = ti.Vector([0.0, 0.0, 1.0])
    c[0] = p
    
    return c

@ti.kernel
def phase_to_color(p: ti.template(), color: ti.template()):
    for i,j in ti.ndrange(n, n):
        for k,l in ti.ndrange(scatter, scatter):
            color[i*scatter+k,j*scatter+l] = get_color(p[ind(i,j)])

@ti.kernel 
def iterate_phase():
    for i,j in ti.ndrange(n,n):
        p_xx = 0.0
        p_yy = 0.0
        laplacian_t = 0.0
        p_x = 0.0
        p_y = 0.0
        p_xy = 0.0
        # those variables above are derivatives of p and t
        
        noise = a * p_old[ind(i,j)] * (1-p_old[ind(i,j)])*(ti.random()-0.5)
        
        if i-1 >= 0:
            p_xx += (p_old[ind(i-1,j)] - p_old[ind(i,j)])
       
        if i+1 < n:
            p_x = p_old[ind(i+1,j)] - p_old[ind(i,j)]
            p_xx += p_x
        else:
            p_x = p_old[ind(i,j)] - p_old[ind(i-1,j)]
        
        if j-1 >= 0:
            p_yy += (p_old[ind(i,j-1)] - p_old[ind(i,j)])
        
        if j+1 < n:
            p_y = p_old[ind(i,j+1)] - p_old[ind(i,j)]
            p_yy += p_y
            p_xy -= p_y
            if i+1 <n:
                p_xy += (p_old[ind(i+1,j+1)] - p_old[ind(i+1,j)])
        else:
            p_y = p_old[ind(i,j)] - p_old[ind(i,j-1)]

        p_x /= dx
        p_y /= dy
        p_xy /= dx*dy
        p_xx /= dx*dx
        p_yy /= dy*dy
       
        laplacian_p = p_xx + p_yy 

        theta = ti.atan2(p_y, p_x)
        cos = ti.cos(J * (theta - theta_0))
        sin = ti.sin(J * (theta - theta_0))
        eps = eps_bar * (1.0 + sigma * cos)
        theta_x = (p_xy * p_x - p_xx * p_y) / (p_x * p_x + p_y * p_y + 1e-10) # we add 1e-10 just in case the denominator is 0
        eps_x = coeff1 * sin * theta_x
        theta_y = (p_yy * p_x - p_xy * p_y) / (p_x * p_x + p_y * p_y + 1e-10)
        eps_y = coeff1 * sin * theta_y 
        g_x = coeff1 * (eps_x*sin*p_y + eps*J*cos*theta_x*p_y + eps*sin*p_xy)
        h_y = coeff1 * (eps_y*sin*p_x + eps*J*cos*theta_y*p_x + eps*sin*p_xy)
        m = alpha * ti.atan2(gamma * (T_eq - t_old[ind(i,j)]), 1.0) / pi

        p_new[ind(i,j)] = p_old[ind(i,j)] + (-p_old[ind(i,j)]*(p_old[ind(i,j)]-1.0)*(p_old[ind(i,j)]-0.5+m) - g_x + h_y + eps*eps*laplacian_p + noise)*dt/tau
        if(p_new[ind(i,j)] < 0):
            p_new[ind(i,j)] = 0
        if(p_new[ind(i,j)] > 1):
            p_new[ind(i,j)] = 1

        b[ind(i,j)] =t_old[ind(i,j)] + kappa*(p_new[ind(i,j)] - p_old[ind(i,j)])


D_builder = ti.SparseMatrixBuilder(n2, n2, max_num_triplets=n2*5)
I_builder = ti.SparseMatrixBuilder(n2, n2, max_num_triplets=n2)

@ti.kernel
def fillDiffusionMatrixBuilder(A: ti.sparse_matrix_builder()):
    for i,j in ti.ndrange(n, n):
        count = 0
        if i-1 >= 0:
            A[ind(i,j), ind(i-1,j)] += 1
            count += 1
        if i+1 < n:
            A[ind(i,j), ind(i+1,j)] += 1
            count += 1
        if j-1 >= 0:
            A[ind(i,j), ind(i,j-1)] += 1
            count += 1
        if j+1 < n:
            A[ind(i,j), ind(i,j+1)] += 1
            count += 1
        A[ind(i,j), ind(i,j)] += -count

@ti.kernel
def fillEyeMatrixBuilder(A: ti.sparse_matrix_builder()):
    for i,j in ti.ndrange(n, n):
        A[ind(i,j), ind(i,j)] += 1


def buildMatrices():
    fillDiffusionMatrixBuilder(D_builder)
    fillEyeMatrixBuilder(I_builder)
    D = D_builder.build()
    I = I_builder.build()
    return D, I

D,I = buildMatrices()
A = I - (dt/(dx**2))*D

def iterate_temperature():
    solver = ti.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(A)
    solver.factorize(A)
    t_new.from_numpy(solver.solve(b))


my_gui = ti.GUI("crystal growth", (res,res))
#my_gui.show()

init()
while my_gui.running:
    iterate_phase()
    iterate_temperature()
    p_old.copy_from(p_new)
    t_old.copy_from(t_new)
    phase_to_color(p_new, pixels)
    my_gui.set_image(pixels)
    my_gui.show()
