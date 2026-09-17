### Arakawa ###

"""

jpp = 
    + f[i-1, j  ] * h[i,   j-1]
    - f[i-1, j  ] * h[i,   j+1]
    - f[i,   j-1] * h[i-1, j  ]
    + f[i,   j-1] * h[i+1, j  ]
    + f[i,   j+1] * h[i-1, j  ]
    - f[i,   j+1] * h[i+1, j  ]
    - f[i+1, j  ] * h[i,   j-1]
    + f[i+1, j  ] * h[i,   j+1]

jpc =
    + f[i-1, j  ] * h[i-1, j-1]
    - f[i-1, j  ] * h[i-1, j+1]
    - f[i,   j-1] * h[i-1, j-1]
    + f[i,   j-1] * h[i+1, j-1]
    + f[i,   j+1] * h[i-1, j+1]
    - f[i,   j+1] * h[i+1, j+1]
    - f[i+1, j  ] * h[i+1, j-1]
    + f[i+1, j  ] * h[i+1, j+1]

jcp = 
    - f[i-1, j-1] * h[i-1, j  ]
    + f[i-1, j-1] * h[i,   j-1]
    + f[i-1, j+1] * h[i-1, j  ]
    - f[i-1, j+1] * h[i,   j+1]
    - f[i+1, j-1] * h[i,   j-1]
    + f[i+1, j-1] * h[i+1, j  ]
    + f[i+1, j+1] * h[i,   j+1]
    - f[i+1, j+1] * h[i+1, j  ]

return (jpp + jpc + jcp) * inv(hx) * inv(hv) / 12

"""
struct Arakawa{DT}
    nx::Int
    nv::Int
    hx::DT
    hv::DT
    factor::DT

    JPP::OffsetArray{Int, 4, Array{Int, 4}}
    JPC::OffsetArray{Int, 4, Array{Int, 4}}
    JCP::OffsetArray{Int, 4, Array{Int, 4}}

    function Arakawa(nx::Int, nv::Int, hx::DT, hv::DT) where {DT}
        JPP = OffsetArray(zeros(Int, 3, 3, 3, 3), -1:+1, -1:+1, -1:+1, -1:+1)
        JPC = OffsetArray(zeros(Int, 3, 3, 3, 3), -1:+1, -1:+1, -1:+1, -1:+1)
        JCP = OffsetArray(zeros(Int, 3, 3, 3, 3), -1:+1, -1:+1, -1:+1, -1:+1)

        JPP[-1, 0, 0, -1] = +1
        JPP[-1, 0, 0, +1] = -1
        JPP[0, -1, -1, 0] = -1
        JPP[0, -1, +1, 0] = +1
        JPP[0, +1, -1, 0] = +1
        JPP[0, +1, +1, 0] = -1
        JPP[+1, 0, 0, -1] = -1
        JPP[+1, 0, 0, +1] = +1

        JPC[-1, 0, -1, -1] = +1
        JPC[-1, 0, -1, +1] = -1
        JPC[0, -1, -1, -1] = -1
        JPC[0, -1, +1, -1] = +1
        JPC[0, +1, -1, +1] = +1
        JPC[0, +1, +1, +1] = -1
        JPC[+1, 0, +1, -1] = -1
        JPC[+1, 0, +1, +1] = +1

        JCP[-1, -1, -1, 0] = -1
        JCP[-1, -1, 0, -1] = +1
        JCP[-1, +1, -1, 0] = +1
        JCP[-1, +1, 0, +1] = -1
        JCP[+1, -1, 0, -1] = -1
        JCP[+1, -1, +1, 0] = +1
        JCP[+1, +1, 0, +1] = +1
        JCP[+1, +1, +1, 0] = -1

        factor = inv(hx) * inv(hv) / 12

        new{DT}(nx, nv, hx, hv, factor, JPP, JPC, JCP)
    end
end

mymod(i, n, w = 1) = abs(i) ≥ n - w ? i - n * sign(i) : i

function (arakawa::Arakawa{DT})(I, J, K) where {DT}
    fi = mymod.(Tuple(J - I), (arakawa.nx, arakawa.nv))
    hi = mymod.(Tuple(K - I), (arakawa.nx, arakawa.nv))

    if any(fi .< -1) || any(fi .> +1) || any(hi .< -1) || any(hi .> +1)
        return zero(DT)
    end

    (arakawa.JPP[fi..., hi...] +
     arakawa.JPC[fi..., hi...] +
     arakawa.JCP[fi..., hi...]) * arakawa.factor
end
