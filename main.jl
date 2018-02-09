#=
構造振動論　レポート
=#
using  Plots, ForwardDiff

#=
define constants...
=#

const H = 1.0e-3
const RHO = 2700
const E = 7.0e+10
const L = 100.0e-3
const xi1 = 4.7300408
const xi2 = 7.8532046
const xi3 = 10.9956078
const xi4 = 14.1571655
const x0 = 0
const pirad = 1.0
const dx = 0.0001
xi = [xi1, xi2, xi3, xi4] # 三角関数係数

#=
define functions...
=#

h_tap(x) = H + 2H * x/L         #tapered beam
betan(xir) = (sinh(xir)+sin(xir)) / (cosh(xir)-cos(xir)) # 解析解双曲線関数係数¥
phir(xir, betaval, x::Vector) = cosh.(xir*x/L*pirad) + cos.(xir*x/L*pirad) - betaval * (sinh.(xir*x/L*pirad) + sin.(xir*x/L*pirad))
@time function ddphir(xir, betaval, x::Vector)
    xrt(x) = cosh.(xir*x/L*pirad) + cos.(xir*x/L*pirad) - betaval * (sinh.(xir*x/L*pirad) + sin.(xir*x/L*pirad))
    dxrt(x) = ForwardDiff.jacobian(xrt,x)        #diagonal matrix
    jacmat = ForwardDiff.jacobian(dxrt,x)
    return sum(jacmat,1)                         #vectorize
end

phiprt(xir, betar, x::Vector) = [phir(xir[1],betar(xir[1]),x),phir(xir[2],betar(xir[2]),x),phir(xir[3],betar(xir[3]),x),phir(xir[4],betar(xir[4]),x)]
ddphiprt(xir, betar, x::Vector) = [ddphir(xir[1],betar(xir[1]),x),ddphir(xir[2],betar(xir[2]),x),ddphir(xir[3],betar(xir[3]),x),ddphir(xir[4],betar(xir[4]),x)]

phitot(Con,phip) = Con[1] * phip[1] + Con[2] * phip[2] + Con[3]  *phip[3] + Con[4] * phip[4]

function main()
    println("Solve tapered beam...")
    kij= zeros(4,4)
    mij = zeros(kij)

    prt(x::Vector) = phiprt(xi, betan, x)
    ddprt(x::Vector) = ddphiprt(xi, betan, x)

    x = collect(x0:dx:L-dx) #区分求積法

    function dm(y::Vector)
        a = prt(y)
        m = zeros(4,4)
        for i in 1:4, j in 1:4
            m[i,j] = dot(a[i], a[j] .* h_tap(x)) * RHO
        end
        return m
    end
    function dk(y::Vector)
        a = ddprt(y)
        k = zeros(4,4)
        for i in 1:4, j in 1:4
            abc = dot(a[i], a[j]' .* h_tap(x).^3)
            k[i,j] = abc[1] * E/12.0
        end
        return k
    end
    kij = dk(x) * dx
    mij = dm(x) * dx

    D, V = eig(*(inv(mij),kij))

    println("Finish Calculation. Now Plotting...")

    pyplot()

    plot(x,prt(x)[1:3])      #一様断面梁のモードプロット

    tp1(x) = phitot(V[1:4,1],phiprt(xi,betan,x))
    tp2(x) = phitot(V[1:4,2],phiprt(xi,betan,x))
    tp3(x) = phitot(V[1:4,3],phiprt(xi,betan,x))
    tp4(x) = phitot(V[1:4,4],phiprt(xi,betan,x))

    plot(x,[tp2(x),tp3(x),tp4(x)])  #テーパー梁のモードプロット

    plot(x,[prt(x)[1],tp2(x)])      #１次モードの比較
    plot(x,[prt(x)[2],tp3(x)])      #２次モードの比較
    plot(x,[prt(x)[3],tp4(x)])      #３次モードの比較

    return D,V
end

ti = time()
D,V = main()
te = time()
println("finish in ", te-ti, "[secs]")
