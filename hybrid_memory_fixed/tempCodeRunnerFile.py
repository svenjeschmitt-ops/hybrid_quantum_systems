
        ax.set(xlabel="Atomic input",ylabel="Joint output",zlabel="Probability",
               zlim=(0,1),title="Retrieval basis populations")
        ax.view_init(elev=25,azim=-53)
        ax.tick_params(labelsize=8)
        ax.text2D(.00,.92,"Input transmon g; cavity vacuum\nOther: all remaining joint states",
                  transform=ax.transAxes,fontsize=8)

        ax=axes[5]