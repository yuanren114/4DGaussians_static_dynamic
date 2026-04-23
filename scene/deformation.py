import torch
import torch.nn as nn
import torch.nn.init as init
from scene.hexplane import HexPlaneField

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.args = args
        self.last_motion_mask = None
        self.last_dx = None
        self.ratio = 0
        self.create_net()

    @property
    def get_aabb(self):
        return self.grid.get_aabb

    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)

    def create_net(self):
        grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4, self.W)]
        else:
            self.feature_out = [nn.Linear(grid_out_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W, self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 16 * 3))
        if self.args.motion_separation:
            self.motion_mask_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))

    def query_time(self, rays_pts_emb, time_emb):
        if self.no_grid:
            hidden = torch.cat([rays_pts_emb[:, :3], time_emb[:, :1]], -1)
        else:
            hidden = self.grid(rays_pts_emb[:, :3], time_emb[:, :1])
        hidden = self.feature_out(hidden)
        return hidden

    @property
    def get_empty_ratio(self):
        return self.ratio

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        zero_time = torch.zeros((rays_pts_emb.shape[0], 1), device=rays_pts_emb.device, dtype=rays_pts_emb.dtype)
        grid_feature = self.grid(rays_pts_emb[:, :3], zero_time)
        hidden = self.feature_out(grid_feature)
        dx = self.pos_deform(hidden)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, time_emb)
        motion_mask = None
        self.last_motion_mask = None
        self.last_dx = None
        if self.args.motion_separation:
            motion_mask = torch.sigmoid(self.motion_mask_deform(hidden))
            self.last_motion_mask = motion_mask

        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
            self.last_dx = torch.zeros_like(pts)
        else:
            dx = self.pos_deform(hidden)
            self.last_dx = dx
            if self.args.motion_separation:
                pts = rays_pts_emb[:,:3] + motion_mask * dx
            else:
                pts = rays_pts_emb[:,:3] + dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)

            if self.args.motion_separation and self.args.motion_gate_rot_scale:
                scales = scales_emb[:,:3] + motion_mask * ds
            else:
                scales = scales_emb[:,:3] + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)

            if self.args.motion_separation and self.args.motion_gate_rot_scale:
                rotations = rotations_emb[:,:4] + motion_mask * dr
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
            opacity = opacity_emb[:,:1] + do
        if self.args.no_dshs:
            shs = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0],16,3])
            shs = shs_emb + dshs

        return pts, scales, rotations, opacity, shs

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    def get_motion_mask(self):
        return self.last_motion_mask
    def get_last_dx(self):
        return self.last_dx
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        self.deformation_net = Deformation(W=net_width, D=defor_depth, args=args)
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.apply(initialize_weights)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        means3D, scales, rotations, opacity, shs = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()
    def get_motion_mask(self):
        return self.deformation_net.get_motion_mask()
    def get_last_dx(self):
        return self.deformation_net.get_last_dx()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb
