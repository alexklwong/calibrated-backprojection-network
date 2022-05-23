classdef GeVrml < Ge
%GEVRML  3D model export into vrml.

% (c) 2007-11-02 Martin Matousek, Czech Technical University in Prague
% Last change: $Date$
%              $Revision$

properties( SetAccess = private, GetAccess = private )
  file % path to vrml file
  fh   % opened vrml file handle
  Xall
  Call
  numX
end

methods

function this = GeVrml( file )
%GEVRML/GEVRML  Constructor.
%  ge = ge_vrml( file )

mkdir_for_file( file )
this.file = file;
this.fh = fopen( file, 'w' );
this.Xall = {};
this.Call = {};
this.numX = 0;

if( this.fh < 0 )
  error( [ 'Cannot wopen file ''' file '''' ] );
end

fprintf( this.fh, '#VRML V2.0 utf8\n' );

end


function close( this )
%GEVRML/CLOSE  Close the VRML file.

if( ~isempty( this.fh ) )
  fclose( this.fh );
  this.fh = [];
end

end

function inx = points( this, X, varargin )
%GEVRML/POINTS  Export of points.
%
%  ge.points( X, ... )

[opt, X] = this.points_arg_helper( X, 'colortype', 'double', varargin{:} );

f = this.fh;

fprintf( f, 'Shape {\n' );
fprintf( f, ' geometry PointSet {\n' );
fprintf( f, '  coord Coordinate {\n' );
fprintf( f, '   point [\n' );
fprintf( f, '%f %f %f,\n', X );
fprintf( f, ']\n}\n' );
if( ~isempty( opt.color ) )
  fprintf( f, ' color Color {\n' );
  fprintf( f, '  color [\n' );
  fprintf( f, '%g %g %g,\n', opt.color );
  fprintf( f, ']\n}\n' );
end
fprintf( f, '}\n}\n' );

inx = this.numX + ( 1:size( X, 2 ) );
this.numX = this.numX + size( X, 2 );
this.Xall{end+1} = X;
if( isempty( opt.color ) )
  this.Call{end+1} = NaN( size( X ) );
else
  this.Call{end+1} = opt.color;
end

end

function faces( this, fi, X, varargin )

f = this.fh;

if( nargin > 2 )
  [opt, X] = this.points_arg_helper( X, ...
                                     'colortype', 'double', ...
                                     varargin{:} );
  C = opt.color;
else
  if( numel( this.Xall ) > 1 )
    this.Xall = { [ this.Xall{:} ] };
    this.Call = { [ this.Call{:} ] };
  end

  key = unique( fi );
  code = nan( 1, max( key ) );
  code( key ) = 1:numel( key );

  X = this.Xall{1}( :, key );
  C = this.Call{1}( :, key );
  fi = code( fi );

  assert( ~any( isnan( fi(:) ) ) )
end

fprintf( f, 'Shape {\n' );
fprintf( f, ' geometry IndexedFaceSet {\n' );
fprintf( f, '  coord Coordinate {\n' );
fprintf( f, '   point [\n' );
fprintf( f, '%f %f %f,\n', X );
fprintf( f, ']\n}\n' );
fprintf( f, '  coordIndex [\n' );
fprintf( f, '%i,%i,%i,-1,\n', fi - 1);
fprintf( f, '  ]\n' );
if( ~isempty( C ) )
  fprintf( f, ' color Color {\n' );
  fprintf( f, '  color [\n' );
  fprintf( f, '    %f %f %f,\n', C );
  fprintf( f, ']\n}\n' );
end
fprintf( f, '}\n}\n' );

end

end % meths.

end % cls.
