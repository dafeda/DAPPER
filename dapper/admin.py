"""Define high-level objects frequently used in DAPPER."""

from dapper import *

class HiddenMarkovModel(NestedPrint):
  """Container for attributes of a Hidden Markov Model (HMM).
  
  This container contains the specification of a "twin experiment",
  i.e. an "OSSE (observing system simulation experiment)".
  """

  def __init__(self,Dyn,Obs,t,X0,**kwargs):
    self.Dyn = Dyn if isinstance(Dyn, Operator)   else Operator  (**Dyn)
    self.Obs = Obs if isinstance(Obs, Operator)   else Operator  (**Obs)
    self.t   = t   if isinstance(t  , Chronology) else Chronology(**t)
    self.X0  = X0  if isinstance(X0 , RV)         else RV        (**X0)

    # Name
    name = inspect.getfile(inspect.stack()[1][0])
    self.name = os.path.relpath(name,'mods/')

    # Kwargs
    abbrevs = {'LP':'liveplotters'}
    for key in kwargs:
      setattr(self, abbrevs.get(key,key), kwargs[key])

    # Defaults
    if not hasattr(self.Obs,"localizer"): self.Obs.localizer = no_localization(self.Nx, self.Ny)
    if not hasattr(self    ,"sectors")  : self.sectors       = {}

    # Validation
    if self.Obs.noise.C==0 or self.Obs.noise.C.rk!=self.Obs.noise.C.M:
        raise ValueError("Rank-deficient R not supported.")
  
  # ndim shortcuts
  @property
  def Nx(self): return self.Dyn.M
  @property
  def Ny(self): return self.Obs.M

  # Print options
  ordering = ['Dyn','Obs','t','X0']


  def simulate(self,desc='Truth & Obs'):
    """Generate synthetic truth and observations."""
    Dyn,Obs,chrono,X0 = self.Dyn, self.Obs, self.t, self.X0

    # Init
    xx    = zeros((chrono.K   +1,Dyn.M))
    yy    = zeros((chrono.KObs+1,Obs.M))

    xx[0] = X0.sample(1)

    # Loop
    for k,kObs,t,dt in progbar(chrono.ticker,desc):
      xx[k] = Dyn(xx[k-1],t-dt,dt) + sqrt(dt)*Dyn.noise.sample(1)
      if kObs is not None:
        yy[kObs] = Obs(xx[k],t) + Obs.noise.sample(1)

    return xx,yy




class Operator(NestedPrint):
  """Container for operators (models)."""
  def __init__(self,M,model=None,noise=None,**kwargs):
    self.M = M

    # None => Identity model
    if model is None:
      model = Id_op()
      kwargs['linear'] = lambda x,t,dt: Id_mat(M)
    self.model = model

    # None/0 => No noise
    if isinstance(noise,RV):
      self.noise = noise
    else:
      if noise is None: noise = 0
      if np.isscalar(noise):
        self.noise = GaussRV(C=noise,M=M)
      else:
        self.noise = GaussRV(C=noise)

    # Write attributes
    for key, value in kwargs.items():
      setattr(self, key, value)
  
  def __call__(self,*args,**kwargs):
    return self.model(*args,**kwargs)

  # Print options
  ordering = ['M','model','noise']

implicit_field = lambda x: dc.field(default=x, repr=False, compare=False)

@dc.dataclass
class _da_defaults:
    """Default kwargs for da_method.

    NB: The da_method's are created at startup.
        => changing this later has no effect."""

    liveplots   : bool = implicit_field(False)
    store_u     : bool = implicit_field(False)
    fail_gently : bool = implicit_field(True)


def da_method(*default_dcs): 
    """Make the decorator that makes the DA classes.

    Example:
    >>> @dc.dataclass
    >>> class ens_defaults:
    >>>   infl          : float = 1.0
    >>>   rot           : bool  = False
    >>> 
    >>> @da_method(ens_defaults)
    >>> class EnKF:
    >>>     N     : int
    >>>     upd_a : str = "Sqrt"
    >>> 
    >>>     def assimilate(self,HMM,xx,yy):
    >>>         ...
    """

    def dataclass_with_defaults(cls):
        """Decorator based on dataclass.

        This adds __init__, __repr__, __eq__, ..., but also includes
        inherited defaults (see stackoverflow.com/a/58130805).

        Also:
         - Wraps assimilate() to provide gentle_fail functionality.
         - Initialises and writes the Stats object.
         - Adds average_stats(), print_averages()."""


        # Default fields invovle: (1) annotations and (2) attributes.
        if not hasattr(cls,'__annotations__'):
            cls.__annotations__ = {}
        def set_field(name,type,val):
            cls.__annotations__[name] = type
            if not isinstance(val,dc.Field):
                val = dc.field(default=val)
            setattr(cls, name, val)

        # APPend default fields without overwriting.
        # Don't implement (by PREpending?) non-default args -- to messy!
        for D in default_dcs + (_da_defaults,):
            # Calling dataclass twice always makes repr=True, so avoid this.
            try:              fields = dc.fields(D)
            except TypeError: fields = dc.fields(dc.dataclass(D))
            for F in fields:
                if F.name not in cls.__annotations__:
                    set_field(F.name,F.type,F)

        # Programmatic defaults (for the class):
        set_field('store_s', bool, implicit_field('Lag' in cls.__annotations__))

        # Create new class (NB: old/new classes have same id) 
        orig_assimilate = cls.assimilate # => store old.assimilate
        cls = dc.dataclass(cls)




        # Shortcut for self.__class__.__name__
        cls.da_method = cls.__name__

        # Add instance methods
        def method(fun):
            setattr(cls,fun.__name__,fun)
            return fun

        @method
        @functools.wraps(orig_assimilate)
        def assimilate(self,HMM,xx,yy,desc=None):
            pb_name_hook = self.da_method if desc is None else desc
            self.stats = Stats(self,HMM,xx,yy)
            call_gently(orig_assimilate,self,HMM,xx,yy)

        @method
        def average_stats(self,free=False):
            """Average (in time) all of the time series in the Stats object.

            If ``free``: del ref to Stats object."""
            self.avrgs = self.stats.average_in_time()
            if free:
                delattr(self,'stats')

        @method
        def print_avrgs(self,keys=()):
            """Tabulated print of averages (those requested by ``keys``)"""
            cfgs = List_of_Configs([self])
            cfgs.print_avrgs(keys)

        method(replay)

        return cls
    return dataclass_with_defaults


class AssimFailedError(RuntimeError):
    pass

def call_gently(fun,self,*args):
    """Wrap fun in try clause to allow execution to continue
    for some types of exceptions."""


    def crop_traceback(ERR,lvl):
        msg = []
        try:
          # If IPython, use its coloring functionality
          __IPYTHON__
          from IPython.core.debugger import Pdb
          import traceback as tb
          pdb_instance = Pdb()
          pdb_instance.curframe = inspect.currentframe() # first frame: this one
          for i, frame_lineno in enumerate(tb.walk_tb(ERR.__traceback__)):
            if i<lvl: continue # skip first frame
            msg += [pdb_instance.format_stack_entry(frame_lineno,context=5)]
        except (NameError,ImportError):
          # No coloring
          msg += ["\n".join(s for s in traceback.format_tb(ERR.__traceback__))]
        return msg

    try:
        fun(self,*args)

    except (AssimFailedError,ValueError,np.linalg.LinAlgError) as ERR:
        if self.fail_gently:
            msg  = ["\n\nCaught exception during assimilation. Traceback:"]
            msg += ["<"*20 + "\n"]
            msg += crop_traceback(ERR,1) + [str(ERR)]
            msg += ["\n" + ">"*20]
            msg += ["Returning stats (time series) object in its current "+\
                "(incompleted) state,\nand resuming program execution.\n"+\
                "Turn off `fail_gently` to fully raise the exception.\n"]
            for s in msg: print(s,file=sys.stderr)
        else: 
            raise ERR

class List_of_Configs(list):
  """List, customized for holding ``da_method`` objects ("configs").

   Modifications to `list`:
   - append() using `+=`, also for single items;
     this is hackey, but convenience is king.
   - append() supports `unique` to avoid duplicates.
   - `__getitem__()` (indexing) that supports lists.
   - searching by attributes: `inds()`.
   - pretty printing (using common/distinct attrs).

   Also:
   - print_averages()
   - gen_names()
   - assimilate()
   """

  def __init__(self,*args,unique=False):
    """Initialize without args, or with a list of configs.
     - unique: if true, then duplicates won't get appended."""
    self.unique = unique
    super().__init__(*args)

  def __iadd__(self,cfg):
    if not hasattr(cfg,'__iter__'):
      cfg = [cfg]
    for item in cfg:
      self.append(item)
    return self

  def append(self,cfg):
    "Append if not unique&present"
    if not (self.unique and cfg in self): super().append(cfg)

  def __getitem__(self, keys):
    """Indexing, also by a list"""
    try:              B=[self[k] for k in keys]   # if keys is list
    except TypeError: B=super().__getitem__(keys) # if keys is int, slice
    if hasattr(B,'__len__'): B = List_of_Configs(B) # Cast
    return B 

  def inds(self,strict=True,**kws):
    """Find (all) indices of configs whose attributes match kws.
     - strict: If True, then configs lacking a requested attribute will match.
    """
    def match(C):
        passthrough = lambda v: 'YOU SHALL NOT PASS' if strict else v
        matches = [getattr(C,k,passthrough(v))==v for k,v in kws.items()]
        return all(matches)

    return [i for i,C in enumerate(self) if match(C)]

  def assimilate(self,HMM,xx,yy,sd=True,free=True,print=False,desc=True):
    "Call config.assimilate() for each config in self."

    if sd is True:
        sd = seed()

    if desc: labels = self.gen_names()
    else:    labels = self.da_methods

    for ic,(label,config) in enumerate(zip(labels,self)):

        # "Variance reduction" (eg. CRN: wikipedia.org/wiki/Variance_reduction)
        # is useful, but should not be relied on for confident conclusions!
        if sd:
            seed(sd)

        config.assimilate(HMM,xx,yy,desc=label)

        config.average_stats(free=free)
        if print:
            config.print_avrgs()

  @property
  def da_methods(self):
    return [config.da_method for config in self]


  def split_attrs(self,nomerge=()):
    """Compile the attributes of the individual configs in the List_of_Confgs,
    and partition them into dicts: distinct, redundant, and common.
    Insert None if attribute not in cfg."""

    def _aggregate_keys():
      "Aggregate keys from all configs"
      if len(self)==0: return []
      # Start with da_method
      aggregate = ['da_method']
      # Aggregate all other keys
      for config in self:
          # Get dataclass fields
          dc_fields = dc.fields(config.__class__)
          dc_names = [F.name for F in dc_fields]
          # For all potential keys:
          for k in config.__dict__.keys():
              # If not already present:
              if k not in aggregate:
                  # If dataclass, check repr:
                  if k in dc_names:
                      if dc_fields[dc_names.index(k)].repr:
                          aggregate.append(k)
                  # Else, just append
                  else:
                      aggregate.append(k)
      # Remove unwanted
      excluded  = [re.compile('^_'),'avrgs','stats','HMM']
      aggregate = filter_out(aggregate,*excluded)
      return aggregate

    distinct, redundant, common = {}, {}, {}

    for key in _aggregate_keys():

        # Want to distinguish actual None's from empty ("N/A").
        # => Don't use getattr(obj,key,None)
        vals = [getattr(config,key,"N/A") for config in self]

        # Sort into distinct, redundant, common
        if key in nomerge:
            # nomerge => Distinct
            dct, vals = distinct, vals
        elif all(vals[0]==v for v in vals):
            # all values equal => common
            dct, vals = common, vals[0]
        else:
            v0 = next(v for v in vals if "N/A"!=v)
            if all(v=="N/A" or v==v0 for v in vals):
            # all values equal or "N/A" => redundant
                dct, vals = redundant, v0
            else:
            # otherwise => distinct
                dct, vals = distinct, vals

        # Replace "N/A" by None
        sub = lambda v: None if v=="N/A" else v
        if isinstance(vals,str): vals = sub(vals)
        else:
            try:                 vals = [sub(v) for v in vals]
            except TypeError:    vals = sub(vals)

        dct[key] = vals
  
    return distinct, redundant, common

  def __repr__(self):
      distinct, redundant, common = self.split_attrs()
      s = '<List_of_Configs> of length %d with attributes:\n'%len(self)
      s += tabulate(distinct)
      s += "\nOther attributes:\n"
      s += str(AlignedDict({**redundant, **common}))
      return s

  @functools_wraps(tabulate_avrgs)
  def _repr_avrgs(self,*args,**kwargs): 
    """Pretty (tabulated) repr of cfgs & avrgs (val±conf)."""
    distinct, redundant, common = self.split_attrs()
  
    # Prepare table components
    headr1, mattr1 = list(distinct.keys()), list(distinct.values())
    headr2, mattr2 = tabulate_avrgs([C.avrgs for C in self],*args,**kwargs,pad='æ')
    # Join 1&2
    headr = headr1 + ['|']             + headr2
    mattr = mattr1 + [['|']*len(self)] + mattr2
  
    table = tabulate(mattr, headr).replace('æ',' ')
    return table
  
  @functools.wraps(_repr_avrgs)
  def print_avrgs(self,*args,**kwargs):
    print(self._repr_avrgs(*args,**kwargs))

  def gen_names(self,abbrev=4,tab=False):
      """Similiar to self.__repr__(), but:
        - returns *list* of names
        - attaches label to each attribute
        - tabulation is only an option
        - abbreviates labels to width abbrev
      """
      distinct, redundant, common = self.split_attrs(nomerge="da_method")
      labels = distinct.keys()
      values = distinct.values()

      # Label abbreviation
      if not abbrev: abbrev = 99
      labels = [(k if len(k)<=abbrev else k[:abbrev-2]+'~'+k[-1]) for k in labels]

      # Make label columns: insert None or lbl+":", depending on value
      column = lambda  lbl,vals: [None if v is None else lbl+":" for v in vals]
      labels = [column(lbl,vals) for lbl, vals in zip(labels,values)]

      # Interlace labels and values
      table = [x for (a,b) in zip(labels,values) for x in (a,b)]

      # Rm da_method label (but keep value)
      table.pop(0)

      # Tabulate
      table = tabulate(table,inds=False, tablefmt="plain")

      # Rm space between lbls/vals
      table = re.sub(':  +',':',table) 

      # Rm alignment
      if not tab:
          table = re.sub(r' +',r' ', table)

      return table.splitlines()




import dill
def save_data(script_name,*args,**kwargs):
  """"Utility for saving experimental data.

  This function uses ``dill`` rather than ``pickle``
  because dill can serialize nearly anything.
  Also, dill automatically uses np.save() for arrays for memory/disk efficiency.

  Takes care of:
   - Path management, using script_name (e.g. script's __file__)
   - Calling dill.dump().
   - Default naming of certain types of arguments.

  Returns filename of saved data. Load namespace dict using
  >>> with open(save_path, "rb") as F:
  >>>     d = dill.load(F)
  """

  def name_args():
      data = {}
      nNone = 0 # count of non-classified objects

      nameable_classes = dict(
            HMM    = lambda x: isinstance(x,HiddenMarkovModel),
            cfgs   = lambda x: isinstance(x,List_of_Configs),
            config = lambda x: hasattr(x,'da_method'),
            stat   = lambda x: isinstance(x,Stats),
            avrg   = lambda x: getattr(x,"_isavrg",False),
          )

      def classify(x):
          for script_name, test in nameable_classes.items():
            if test(x): return script_name
          # Defaults:
          if isinstance(x,list): return "list"
          else:                  return None

      for x in args:
          Class = classify(x)

          if Class == "list":
            Class0 = classify(x[0])
            if Class0 in nameable_classes and all([Class0==classify(y) for y in x]):
              Class = Class0 + "s" # plural
            else:
              Class = None

          elif Class is None:
            nNone += 1
            Class = "obj%d"%nNone

          data[Class] = x
      return data

  filename  = save_dir(script_name,host=False) + "run_"
  filename += str(1 + max(get_numbering(filename),default=0)) + ".pickle"
  print("Saving data to",filename)

  with open(filename,"wb") as filestream:
      dill.dump({**kwargs, **name_args()}, filestream)

  return filename





