from icecube import tableio, dataclasses, MuonGun
from icecube import simclasses, clsim, phys_services
from icecube.dataclasses import I3Particle

# I3MCTree
import numpy as np


def make_vector(direction):
    return np.array([direction.x, direction.y, direction.z])


def get_radius(axis, track):
    r = np.subtract(make_vector(track.pos), make_vector(axis.pos))
    l = np.inner(make_vector(axis.dir), r)
    return np.sqrt(max(0, np.inner(r, r) - l * l))


def get_depth_km(z):
    return (
        dataclasses.I3Constants.SurfaceElev - dataclasses.I3Constants.OriginElev - z
    ) / 1000


class MCPrimaryConverter(tableio.I3Converter):
    # class MCPrimaryConverter(dataclasses.converters.I3ParticleConverter ):

    """converter to store minorID and majorID of primary particles"""
    # booked = dataclasses.I3MCTree
    booked = dataclasses.I3Particle

    def __init__(
        self,
        key_name=None,
        count_photons={},
        filter_names=[],
        add_nus=False,
        max_muon_multiplicity=10,
        add_shower_muons=False,
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        key_name : str
            name of the I3MCTree to use as source for primary particles
        count_photons : dict
            if given, count the number of photons produced by the primary particles, e.g.
            count_photons = dict(key="I3MCTree_sliced",
                                n_photons_per_step=1000,
            )
        filter_names : list
            if given, add the filter names to the table

        """

        super().__init__(*args, **kwargs)
        # self.converter = dataclasses.converters.I3ParticleConverter(*args, **kwargs)
        self.key_name = key_name
        self.filter_names = filter_names
        self.count_photons = count_photons
        self.add_nus = add_nus
        self.add_shower_muons = add_shower_muons
        self.max_muon_multiplicity = max_muon_multiplicity
        if self.count_photons:
            self.count_photons.setdefault("n_photons_per_step", 1000)
            self.count_photons.setdefault("key", "I3MCTree_sliced")
        if self.max_muon_multiplicity:

            self.surface = MuonGun.Cylinder(1600, 800)


    def CreateDescription(self, part):
        desc = tableio.I3TableRowDescription()
        desc.add_field("energy", tableio.types.Float64, "GeV", "Energy")
        desc.add_field("minorID", tableio.types.Float64, "", "Minor ID")
        desc.add_field("majorID", tableio.types.Float64, "", "Major ID")
        desc.add_field("azimuth", tableio.types.Float64, "radian", "azimuth")
        desc.add_field("zenith", tableio.types.Float64, "radian", "zenith")
        desc.add_field("theta", tableio.types.Float64, "radian", "theta")
        desc.add_field("x", tableio.types.Float64, "m", "x")
        desc.add_field("y", tableio.types.Float64, "m", "y")
        desc.add_field("z", tableio.types.Float64, "m", "z")
        desc.add_field("type", tableio.types.Float64, "", "type")
        desc.add_field("length", tableio.types.Float64, "m", "Length")
        desc.add_field("pdg_encoding", tableio.types.Float64, "", "PDG ID")

        if self.add_nus:
            desc.add_field("nu1_energy", tableio.types.Float64, "", "nu1_energy")
            desc.add_field("nu2_energy", tableio.types.Float64, "", "nu2_energy")
            desc.add_field("n_nu", tableio.types.Float64, "", "n_nu")

        if self.add_shower_muons:
            desc.add_field("shower_mu1_energy", tableio.types.Float64, "", "shower_mu1_energy")
            desc.add_field("shower_mu2_energy", tableio.types.Float64, "", "shower_mu2_energy")
            desc.add_field("n_shower_mu", tableio.types.Float64, "", "n_shower_muons")

        if self.max_muon_multiplicity:
            desc.add_field(
                "multiplicity", tableio.types.Float64, "", "muon multiplicity"
            )
            desc.add_field("depth", tableio.types.Float64, "km", "depth")
            desc.add_field("cos_theta", tableio.types.Float64, "", "cos_theta")
            for i in range(1, self.max_muon_multiplicity + 1):
                desc.add_field(
                    f"mu{i}_energy", tableio.types.Float64, "", f"mu{i} energy"
                )
                desc.add_field(
                    f"mu{i}_radius", tableio.types.Float64, "m", f"mu{i} radius"
                )
                for attr in ['pos', 'dir']:
                    for comp in ['x', 'y', 'z']:
                        desc.add_field(
                            f"mu{i}_{attr}_{comp}", tableio.types.Float64, "m", f"mu{i} {attr} {comp}"
                        )

                    


        desc.add_field(
            "interaction_height",
            tableio.types.Float64,
            "",
            "Corsika interaction height",
        )
        if self.count_photons:
            desc.add_field("n_photons", tableio.types.Float64, "", "number of photons")
            self.photon_counter = PhotonCounter(
                photonsPerStep=self.count_photons["n_photons_per_step"]
            )

        if self.filter_names:
            for filter_name in self.filter_names:
                desc.add_field(filter_name, tableio.types.Int32, "", "")
                # desc.add_field(filter_name + "_condition", tableio.types.Int32, "", "")
                # desc.add_field(filter_name + "_prescale", tableio.types.Int32, "", "")

        return desc

    def Convert(self, particle, row, frame):
        """ """
        if self.key_name:
            primary = frame[self.key_name].primaries[0]
        else:
            primary = particle

        keys = [
            "energy",
            "pdg_encoding",
            "length",
            "type",
        ]

        for key in keys:
            row[key] = getattr(primary, key)

        row["minorID"] = primary.id.minorID
        row["majorID"] = primary.id.majorID
        row["x"], row["y"], row["z"] = primary.pos
        row["azimuth"] = primary.dir.azimuth
        row["zenith"] = primary.dir.zenith
        row["theta"] = primary.dir.theta
        row["interaction_height"] = frame["CorsikaInteractionHeight"].value

        if self.count_photons:
            row["n_photons"] = self.photon_counter.count_photons_in_frame(
                frame, key=self.count_photons["key"]
            )

        if self.filter_names:
            filter_key = "QFilterMask"
            if filter_key in frame:
                filter_mask = frame[filter_key]
                for filter_name in self.filter_names:
                    row[filter_name] = bool(filter_mask[filter_name])

        if self.add_nus:
            nu_types = [
                I3Particle.NuMu,
                I3Particle.NuE,
                I3Particle.NuTau,
                I3Particle.NuMuBar,
                I3Particle.NuEBar,
                I3Particle.NuTauBar,
            ]
            nus = [p for p in frame[self.key_name] if p.type in nu_types]
            nus = sorted(nus, key=lambda x: x.energy, reverse=True)
            row["n_nu"] = len(nus)

            row["nu1_energy"] = 0 if len(nus) < 1 else nus[0].energy
            row["nu2_energy"] = 0 if len(nus) < 2 else nus[1].energy

        if self.add_shower_muons:
            from_mmctracks = False
            key = "I3MCTree"
            mu_types = [I3Particle.MuMinus, I3Particle.MuPlus]
            if key ==   "MMCTrackList":
                shower_muons = [p.GetI3Particle() for p in frame[key]]
                shower_muons = [p for p in shower_muons if p.pdg_encoding in mu_types]
                shower_muons = sorted(shower_muons, key=lambda x: x.energy, reverse=True)
                row["n_shower_mu"] = len(shower_muons)
            elif key == "I3MCTree":

                shower_muons = [p for p in frame[key] if p.type in mu_types]
                shower_muons = sorted(shower_muons, key=lambda x: x.energy, reverse=True)
                row["n_shower_mu"] = len(shower_muons)

            row["shower_mu1_energy"] = 0 if len(shower_muons) < 1 else shower_muons[0].energy
            row["shower_mu2_energy"] = 0 if len(shower_muons) < 2 else shower_muons[1].energy
            # print([k.energy for k in shower_muons])

        if self.max_muon_multiplicity:
            max_multiplicity = self.max_muon_multiplicity
            muons_at_surface = MuonGun.muons_at_surface(frame, self.surface)
            primary_steps = self.surface.intersection(primary.pos, primary.dir)
            row["multiplicity"] = len(muons_at_surface)
            row["depth"] = get_depth_km(
                primary.pos.z + primary_steps.first * primary.dir.z
            )
            row["cos_theta"] = np.cos(primary.dir.zenith)

            for itrack in range(1, max_multiplicity + 1):
                row[f"mu{itrack}_energy"] = 0
                row[f"mu{itrack}_radius"] = 0

            for itrack, track in enumerate(muons_at_surface, 1):
                if itrack > max_multiplicity:
                    break
                row[f"mu{itrack}_energy"] = track.energy
                row[f"mu{itrack}_radius"] = get_radius(primary, track)
                for attr in ['pos', 'dir']:
                    for comp in ['x', 'y', 'z']:
                        row[f"mu{itrack}_{attr}_{comp}"] = getattr(getattr(track, attr ), comp)
                        # print(itrack, attr, comp, getattr(getattr(track, attr ), comp))

        ##########
        ##########

        # if self.eval_model:
        #     print("ROW:", row)
        #     print(dir(row))
        #     import pandas as pd
        #     import sys

        #     print(sys.path)
        #     tf_path = "/home/navidkrad/venvs/py3.11_tf/lib/python3.11/site-packages"
        #     if tf_path not in sys.path:
        #         print("WARNING ADDING TF ENV TO SYS.PATH")
        #         # assert False
        #         sys.path = [tf_path] + sys.path
        #     # from ModelFactory import ModelFactory
        #     from utils import preproc

        #     for k in row.keys():
        #         print(k)
        #         print(row[k])
        #     di = {k: row[k] for k in row.keys()}
        #     print(di)
        #     df_row = pd.Series(di).to_frame().T
        #     print(df_row)
        #     print(preproc(df_row))

        # print(pd.DataFrame({k: row[k] for k in row.keys()}))
        # print(list(row))

        #############
        #############

        return 1


class SimpleFrameConverter(tableio.I3Converter):

    # booked = dataclasses.I3Particle
    booked = dataclasses.I3Particle

    def __init__(self, keys=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys

    def CreateDescription(self, frame):
        desc = tableio.I3TableRowDescription()
        for key in self.keys:
            desc.add_field(key, tableio.types.Float64, "", key)
        return desc

    def Convert(self, particle, row, frame):
        print(frame.keys())
        for key in self.keys:
            if key in frame:
                row[key] = frame[key].value
            else:
                print('key not found %s'%key)
        return 1



class PhotonCounter:
    def __init__(self, photonsPerStep=200):
        converter = clsim.I3CLSimLightSourceToStepConverterPPC(
            photonsPerStep=photonsPerStep
        )
        converter.SetWlenBias(clsim.GetIceCubeDOMAcceptance())
        converter.SetMediumProperties(clsim.MakeIceCubeMediumProperties())
        converter.SetRandomService(phys_services.I3GSLRandomService(0))
        converter.Initialize()

        self.converter = converter
        # self.__photonsPerStep = photonsPerStep
        self.photonsPerStep = photonsPerStep
        self.id = 0

    def enqueue_source(self, source):
        if isinstance(source, simclasses.I3LightSource):
            pass
        else:
            source = simclasses.I3LightSource(source)
        self.converter.EnqueueLightSource(source, self.id)
        self.id += 1

    def count_steps(self):
        steps_all = []
        while self.converter.MoreStepsAvailable():
            steps = self.converter.GetConversionResult()
            steps_all.append(len(steps))
        return sum(steps_all)

    def count_photons(self):
        n_steps = self.count_steps()
        # print(n_steps, self.photonsPerStep)
        return n_steps * self.photonsPerStep

    def count_photons_in_frame(self, frame, key="I3MCTree"):
        mct = frame[key]
        n_photons = []

        if self.converter.MoreStepsAvailable():
            raise RuntimeError(
                "apparently conversion was not finalized in previous calls...."
            )

        for primary in mct.primaries:
            daughters = mct.get_daughters(primary)
            for mu in daughters:
                if mu.type not in [mu.MuMinus, mu.MuPlus]:
                    continue
                mu_segments = []
                for d in mct.get_daughters(mu):
                    if d.type in (d.NuE, d.NuMu, d.NuTau, d.NuEBar, d.NuMuBar, d.NuTauBar):
                        continue
                    if (d.type == mu.type):
                        continue
                    if (d.location_type == d.InIce) and (d.shape != d.Dark):
                        self.enqueue_source(d)

        n_photons.append(self.count_photons())
        return sum(n_photons)
