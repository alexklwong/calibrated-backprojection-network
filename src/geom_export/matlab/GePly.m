classdef GePly < Ge
%GEPLY  3D model export into PLY file.

% (c) 2017-01-24 Martin Matousek, Czech Technical University in Prague
% Last change: $Date: 2017-04-10 19:40:41 +0200$
%              $Revision: 74a104c$

properties( SetAccess = private, GetAccess = private )
  file      % path to ply file
  fh        % opened ply file handle
  binary    % PLY format: true = binary, false = ascii
  vertices  % cell array of numeric subarrays of vertices
  colors    % cell array of numeric subarrays of corresponding vertex colours
  vcount    % total count of vertices
  vfaces    % cell array of numeric subarrays of faces (v. indices)
end

methods

function this = GePly( file, what )
%GEPLY/GEPLY  Constructor.
%  ge = ge_ply( file [, format ] )
%
%    format - 'binary' (default) or 'ascii'

mkdir_for_file( file )
this.fh = fopen( file, 'w' );
this.vertices = {};
this.vcount = 0;
this.colors = {};
this.vfaces = {};

if( this.fh < 0 )
  error( [ 'Cannot wopen file ''' file '''' ] );
end

this.file = file;

if( nargin == 1 ), what = 'binary'; end

switch( what )
  case 'ascii'
    fprintf( this.fh, 'ply\nformat ascii 1.0\n' );
    this.binary = false;
  case 'binary'
    fprintf( this.fh, 'ply\nformat binary_little_endian 1.0\n' );
    this.binary = true;
  otherwise
    error( 'Unknown PLY format ''%s''.', what );
end

end

function close( this )
%GEPLY/CLOSE  Finish and close the PLY file.

if( ~isempty( this.fh ) )
  % all vertices and faces
  v = [ this.vertices{:} ];
  f = [ this.vfaces{:} ];

  % if there are some colors, propagate single and default colors to points
  if( all( cellfun( @isempty, this.colors ) ) )
    c = uint8( [] );
  else
    for i = 1:numel( this.colors )
      n = size( this.vertices{i}, 2 );
      if( isempty( this.colors{i} ) )
        this.colors{i} = ones( 3, n, 'uint8' ) * 255;
      elseif( size( this.colors{i}, 2 ) == 1 )
        this.colors{i} = this.colors{i}( :, ones( 1, n ) );
      end
    end

    c = [ this.colors{:} ];
    assert( isequal( size( c ), size( v ) ) );
  end

  % write head (colours are used only if needed)
  fprintf( this.fh, 'element vertex %i\n', size( v, 2 ) );
  fprintf( this.fh, 'property float x\n' );
  fprintf( this.fh, 'property float y\n' );
  fprintf( this.fh, 'property float z\n' );
  if( ~isempty(c) )
    fprintf( this.fh, 'property uchar red\n' );
    fprintf( this.fh, 'property uchar green\n' );
    fprintf( this.fh, 'property uchar blue\n' );
  end

  fprintf( this.fh, 'element face %i\n', size( f, 2 ) );
  fprintf( this.fh, 'property list uchar int vertex_indices\n' );
  fprintf( this.fh, 'end_header\n' );

  % write data
  if( this.binary )
    % vertices
    v = typecast( v(:), 'uint8' );
    v = reshape( v, 12, [] );
    v = [ v; c];
    fwrite( this.fh, v );

    % faces (always three vertex indices)
    if( ~isempty( f ) )
      f = typecast( f(:), 'uint8' );
      f = reshape( f, 12, [] );
      f = [ ones( 1, size( f, 2 ), 'uint8' ) * 3; f ];
      fwrite( this.fh, f );
    end
  else
    % vertices
    if( isempty( c ) )
      fprintf( this.fh, '%f %f %f\n', v );
    else
      fprintf( this.fh, '%f %f %f %i %i %i\n', [ v; single(c) ] );
    end

    % faces (always three vertex indices)
    if( ~isempty( v ) )
      fprintf( this.fh, '3 %i %i %i\n', f );
    end
  end

  fclose( this.fh );
  this.fh = [];
end

end

function inx = points( this, X, varargin )

[opt, X] = this.points_arg_helper( X, 'colortype', 'uint8', varargin{:} );

if( nargout > 0 )
  inx = this.vcount + (1:size( X, 2 ));
end

this.vcount = this.vcount + size( X, 2 );

this.vertices = [ this.vertices {single(X)} ];
this.colors = [ this.colors { opt.color } ];
end

function faces( this, f, varargin )
%GEPLY/FACES  Export of faces.
%
%  ge.faces( f, ... )
%    f - 3xn indices of points

if( size( f, 1 ) ~= 3 )
  error( 'faces must have exactly 3 vertices.' )
end

if( nargin > 2 )
  inx = this.points( varargin{:} );
  f = f + inx(1) - 1;
end

if( any( f(:) < 1 ) || any( f(:) > this.vcount ) )
  error( 'Face contains wrong vertex indices.' );
end

this.vfaces = [ this.vfaces { uint32( f-1 ) } ]; % 0 based

end

end % meths.

end % cls.
